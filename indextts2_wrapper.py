"""IndexTTS2 Python API 封装（v0.4.3+ 引入）。

封装 ``indextts.infer_v2.IndexTTS2`` 的构造 / 加载 / 推理 / 卸载，
提供线程安全的模型管理接口。

与 VoxCPMWrapper 的关键差异：
    1. IndexTTS2 没有 ``from_pretrained`` 类方法，模型在 ``__init__`` 中直接加载
    2. ``infer()`` 写文件（``output_path``），不返回 numpy 数组 → 需用
       ``soundfile.read()`` 读回 numpy
    3. 支持 4 种情感控制模式（参考音频 / 8 维向量 / 文本描述 / 无情感）
    4. 需 monkey-patch ``torch.load``（PyTorch 2.6+ ``weights_only=True`` 兼容）
    5. 辅助模型来自 4 个独立 HuggingFace 仓库，通过 ``aux_paths`` 参数注入

规范：
    - project-rules.md §第二部分 §2：load() 禁止兜底下载
    - project-rules.md §第二部分 §3：torch.load monkey-patch 必须 try/finally 包裹
    - project-rules.md §第二部分 §4：device="auto" 在 load() 入口统一解析
    - PR/development-guide.md §28.20A.3：IndexTTS2Wrapper 封装规范

参考：``engines/voxcpm_tts_c4a8f2e7/voxcpm_wrapper.py``
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import EngineConfig

logger = logging.getLogger(__name__)

# v0.3.5+ 设备策略合法取值（平台注入 AIRELIABLE_DEVICE_POLICY 环境变量）
#   auto    ：引擎自行决定（现状行为：显存预检不足时抛错）
#   gpu     ：强制全 GPU，显存不足直接报错（与 auto 的预检行为一致）
#   gpu+cpu ：GPU 优先，显存不足时警告 + 降级 CPU 加载
#   cpu     ：强制 CPU（跳过显存预检）
DEVICE_POLICIES = ("auto", "gpu", "gpu+cpu", "cpu")


def _read_device_policy() -> str:
    """读取设备策略（v0.3.5+，平台注入 AIRELIABLE_DEVICE_POLICY）。"""
    policy = (os.environ.get("AIRELIABLE_DEVICE_POLICY", "auto").strip() or "auto")
    if policy not in DEVICE_POLICIES:
        logger.warning("非法设备策略 %r（合法值：%s），回退 auto", policy, DEVICE_POLICIES)
        return "auto"
    return policy


# ===========================================================================
# 常量
# ===========================================================================

#: IndexTTS2 默认采样率（需从 config.yaml 的 s2mel.preprocess_params.sr 确认）
_DEFAULT_SAMPLE_RATE = 24000

#: FP16 模式下预估所需显存（GB），用于加载前预检
_REQUIRED_VRAM_GB_FP16 = 6.0

#: 主模型文件清单（7 个文件，来自 IndexTeam/IndexTTS-2）
#: (filename, min_size_bytes) —— min_size 用于防止残文件
#:
#: v0.2.16+ 修订：情感/说话人矩阵文件名修正
#:     config.yaml 中 ``emo_matrix: feat2.pt`` / ``spk_matrix: feat1.pt``
#:     IndexTTS2 库通过 ``self.cfg.emo_matrix`` / ``self.cfg.spk_matrix`` 读取路径，
#:     实际加载的是 ``feat2.pt`` / ``feat1.pt``（HuggingFace 仓库实际文件名），
#:     而非旧 manifest 错误声明的 ``emo_matrix.pt`` / ``spk_matrix.pt``（仓库中不存在 → HTTP 404）。
_MAIN_MODEL_FILES: list[tuple[str, int]] = [
    ("config.yaml", 1024),            # 主模型配置（标志文件）
    ("bpe.model", 1024),              # BPE tokenizer
    ("gpt.pth", 1024 * 1024),        # GPT 自回归模型权重（~1.5GB）
    ("s2mel.pth", 1024 * 1024),      # s2mel flow 模型权重（~800MB）
    ("wav2vec2bert_stats.pt", 1024), # w2v-bert 特征统计
    ("feat2.pt", 1024),              # 情感矩阵（config.yaml: emo_matrix: feat2.pt）
    ("feat1.pt", 1024),              # 说话人矩阵（config.yaml: spk_matrix: feat1.pt）
]

#: Qwen 情感分析子模型文件清单（v0.2.16+ 新增）
#: IndexTTS2 库通过 ``self.qwen_emo = QwenEmotion(os.path.join(self.model_dir, self.cfg.qwen_emo_path))``
#: 加载 config.yaml 的 ``qwen_emo_path: qwen0.6bemo4-merge/`` 子目录下的 Qwen 模型。
#: 11 个文件来自 IndexTeam/IndexTTS-2 仓库的 ``qwen0.6bemo4-merge/`` 子目录，
#: ``model.safetensors`` 是 1.19GB 大权重，其余是 tokenizer/config 等小文件。
#: 相对路径相对于 ``{variant_path}/qwen0.6bemo4-merge/``。
_QWEN_EMO_FILES: list[tuple[str, int]] = [
    # v0.3.5+ 修订：config.json / tokenizer_config.json 实测可能只有几百字节
    # （如 IndexTeam/IndexTTS-2 的 config.json=727 字节），阈值从 1024 降到 100
    # 与同类型小配置文件（generation_config / added_tokens / special_tokens_map）一致，
    # 仍能防御 0 字节残文件（断点续传未完成遗留）。
    ("config.json", 100),
    ("generation_config.json", 100),
    ("model.safetensors", 1024 * 1024 * 100),  # ~1.19GB
    ("tokenizer.json", 1024 * 1024),            # ~11MB
    ("tokenizer_config.json", 100),             # 实测 5433 字节，但同类型小文件阈值
    ("vocab.json", 1024 * 1024),                 # ~2.6MB
    ("merges.txt", 1024 * 1024),                 # ~1.6MB
    ("special_tokens_map.json", 100),
    ("added_tokens.json", 100),
    ("chat_template.jinja", 100),
    ("Modelfile", 100),
]

#: 辅助模型文件清单（4 个独立 HF 仓库，存放在 hf_cache/ 子目录）
#: key = 辅助模型名称，value = (相对路径列表, min_size_bytes)
#: 相对路径相对于 {variant_path}/hf_cache/
_AUX_MODEL_FILES: dict[str, list[tuple[str, int]]] = {
    # 1. w2v-bert-2.0 整仓（facebook/w2v-bert-2.0，~2.3GB）
    # v0.2.16+ 修订：HF 仓库实际只有 4 个文件（无 tokenizer），
    # Wav2Vec2BertModel.from_pretrained 只需 config.json + preprocessor_config.json +
    # model.safetensors，旧 manifest 声明的 pytorch_model.bin / tokenizer 等文件不存在。
    "w2v_bert": [
        ("w2v-bert-2.0/config.json", 1024),
        ("w2v-bert-2.0/preprocessor_config.json", 100),
        ("w2v-bert-2.0/model.safetensors", 1024 * 1024 * 100),  # ~2.3GB
    ],
    # 2. semantic_codec 单文件（amphion/MaskGCT）
    "semantic_codec": [
        ("semantic_codec_model.safetensors", 1024 * 1024),
    ],
    # 3. campplus 单文件（funasr/campplus）
    "campplus": [
        ("campplus_cn_common.bin", 1024 * 1024),
    ],
    # 4. bigvgan 整仓（nvidia/bigvgan_v2_22khz_80band_256x）
    "bigvgan": [
        ("bigvgan/config.json", 1024),
        ("bigvgan/bigvgan_generator.pt", 1024 * 1024),
    ],
}

#: aux_paths 字段 → HF 仓库 ID 映射（用于错误提示）
_AUX_REPO_MAP: dict[str, str] = {
    "w2v_bert": "facebook/w2v-bert-2.0",
    "semantic_codec": "amphion/MaskGCT",
    "campplus": "funasr/campplus",
    "bigvgan": "nvidia/bigvgan_v2_22khz_80band_256x",
}


# ===========================================================================
# IndexTTS2Wrapper
# ===========================================================================

class IndexTTS2Wrapper:
    """IndexTTS2 模型封装器。

    封装 ``indextts.infer_v2.IndexTTS2`` 的 Python API，提供：
        - 线程安全的模型加载/卸载
        - 5 种合成模式的参数构造（v0.1 仅前 3 种）
        - 设备选择（auto/cpu/cuda）
        - 文件完整性校验（不下载，只校验）
        - torch.load monkey-patch（PyTorch 2.6+ 兼容）

    关键约束（project-rules.md §第二部分 §2）：
        - ``load()`` 只负责加载，不负责下载
        - 文件缺失时直接抛 ``RuntimeError``，让用户去模型管理界面下载
        - 禁止调用 ``snapshot_download()`` / ``hf_hub_download()`` 等网络下载函数
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self._model = None
        self._lock = threading.Lock()
        self._loaded_variant: Optional[str] = None
        self._sample_rate: int = _DEFAULT_SAMPLE_RATE
        # v0.3.5+ 设备策略（平台注入 AIRELIABLE_DEVICE_POLICY）
        self._device_policy: str = _read_device_policy()
        # v0.2.5+ 启动时立即解析真实设备，避免 /status 在未加载模型时误报 "device=cpu"
        self._device: str = self._resolve_device(config.device)
        # v0.3.5+ cpu 策略强制 CPU（优先于 auto 检测与 config.device）
        if self._device_policy == "cpu":
            self._device = "cpu"
        self._device_name: str = self._get_device_name(self._device)
        logger.info(
            "IndexTTS2Wrapper 初始化: config.device=%s, resolved_device=%s, device_name=%s, device_policy=%s",
            config.device, self._device, self._device_name, self._device_policy,
        )

    # ===================================================================
    # 公开属性
    # ===================================================================

    @property
    def device_name(self) -> str:
        """设备显示名称（如 "NVIDIA GeForce RTX 4090" / "CPU"）。"""
        return self._device_name

    @property
    def is_loaded(self) -> bool:
        """模型是否已加载。"""
        return self._model is not None

    @property
    def loaded_variant(self) -> Optional[str]:
        """已加载的变体名。"""
        return self._loaded_variant

    @property
    def sample_rate(self) -> int:
        """输出采样率。"""
        return self._sample_rate

    @property
    def device(self) -> str:
        """当前推理设备。"""
        return self._device

    # ===================================================================
    # load / unload
    # ===================================================================

    def load(self, variant_name: str, variant_path: str, device: str = "auto") -> None:
        """加载 IndexTTS2 模型。

        流程（project-rules.md §第二部分 §2.2 正确架构）：
            1. 文件完整性校验（主模型 7 文件 + 辅助模型 4 仓库，不下载）
            2. 显存预检（FP16 需 ~6GB）
            3. torch.load monkey-patch（try/finally 包裹整个加载流程）
            4. 构造 aux_paths（指向 hf_cache 子目录，绕过 ensure_models_available）
            5. 调用 ``IndexTTS2.__init__()`` 加载模型
            6. 读取实际采样率（从 config.yaml 或模型属性）

        Args:
            variant_name: variant 名称（如 "IndexTeam/IndexTTS-2"），用于状态查询
            variant_path: 模型目录路径（如 ``models/voice/tts/indextts2/IndexTTS-2``）
            device: 推理设备（auto/cpu/cuda:0）

        Raises:
            RuntimeError: 文件缺失 / 显存不足 / 加载失败

        v0.2.5+ 修复：``_loaded_variant`` 存 variant 名而非路径，避免 UI 层比较时
        永远不相等导致每次推理都触发 unload + load。
        """
        with self._lock:
            if self._model is not None:
                logger.warning("模型已加载（variant=%s），先卸载", self._loaded_variant)
                self._unload_unsafe()

            # 1. 解析设备（project-rules.md §第二部分 §4：auto → cuda:0 / cpu）
            self._device = self._resolve_device(device)
            # v0.3.5+ cpu 策略强制 CPU（优先于 auto 检测与显式 device 参数）
            if self._device_policy == "cpu":
                self._device = "cpu"
            self._device_name = self._get_device_name(self._device)
            logger.info(
                "加载 IndexTTS2 模型: name=%s, path=%s, device=%s, device_policy=%s",
                variant_name, variant_path, self._device, self._device_policy,
            )

            # 2. 文件完整性校验（不下载，只校验）
            self._check_files_integrity(variant_path)

            # 3. 显存预检（v0.3.5+ gpu+cpu 策略：不足时降级 CPU 而非抛错）
            if not self._check_vram_available(self._device, required_gb=_REQUIRED_VRAM_GB_FP16):
                logger.warning(
                    "gpu+cpu 策略显存不足，IndexTTS2 模型降级 CPU 加载（推理速度会显著变慢）"
                )
                self._device = "cpu"
                self._device_name = self._get_device_name("cpu")

            # 4. monkey-patch torch.load（project-rules.md §第二部分 §3）
            #    PyTorch 2.6+ 默认 weights_only=True，会拒绝加载含 numpy 标量的权重
            #    patch 必须用 try/finally 包裹整个 load 流程，覆盖所有子模型加载
            import torch

            _orig_torch_load = torch.load

            def _patched_torch_load(*args, **kwargs):
                kwargs.setdefault("map_location", "cpu")
                kwargs.setdefault("weights_only", False)
                return _orig_torch_load(*args, **kwargs)

            # 5. 加载模型
            try:
                torch.load = _patched_torch_load
                logger.debug("torch.load monkey-patch 已启用（weights_only=False, map_location=cpu）")

                from indextts.infer_v2 import IndexTTS2

                # 构造 aux_paths（指向 hf_cache 子目录，绕过 ensure_models_available 自动下载）
                aux_paths = self._resolve_aux_paths(variant_path)
                logger.info("aux_paths 注入: %s", {k: str(v) for k, v in aux_paths.items()})

                cfg_path = str(Path(variant_path) / "config.yaml")
                use_fp16 = self._device.startswith("cuda")

                self._model = IndexTTS2(
                    cfg_path=cfg_path,
                    model_dir=variant_path,
                    use_fp16=use_fp16,
                    device=self._device,
                    use_cuda_kernel=False,      # 默认关闭，需用户显式开启
                    use_deepspeed=False,        # 同上
                    use_accel=False,            # GPT2 flash-attention，同上
                    use_torch_compile=False,    # torch.compile 优化 s2mel，同上
                    aux_paths=aux_paths,
                )

                # v0.2.5+ 必须存 variant 名（如 "IndexTeam/IndexTTS-2"），禁止存路径
                self._loaded_variant = variant_name

                # 读取实际采样率（从 config.yaml 的 s2mel.preprocess_params.sr）
                self._sample_rate = self._read_sample_rate(variant_path)

                logger.info(
                    "IndexTTS2 模型加载成功: variant=%s, sample_rate=%d, device=%s, fp16=%s",
                    variant_name, self._sample_rate, self._device, use_fp16,
                )
            except RuntimeError:
                # 文件校验 / 显存预检抛出的友好错误，直接向上抛
                self._model = None
                self._loaded_variant = None
                raise
            except Exception as e:
                self._model = None
                self._loaded_variant = None
                err_str = str(e)
                # OOM 错误友好化
                if "CUDA out of memory" in err_str or "显存不足" in err_str:
                    raise RuntimeError(
                        "GPU 显存不足，无法加载 IndexTTS2 模型。"
                        "建议：1) 关闭其他占用显存的程序 2) 切换到 CPU 模式 3) 卸载其他已加载的引擎"
                    ) from e
                logger.exception("IndexTTS2 模型加载失败")
                raise RuntimeError(f"IndexTTS2 模型加载失败: {e}") from e
            finally:
                torch.load = _orig_torch_load
                logger.debug("torch.load monkey-patch 已恢复")

    def unload(self) -> None:
        """卸载模型，释放显存。"""
        with self._lock:
            self._unload_unsafe()

    def _unload_unsafe(self) -> None:
        """卸载模型（无锁版本）。"""
        if self._model is None:
            return

        logger.info("卸载 IndexTTS2 模型: variant=%s", self._loaded_variant)
        self._model = None
        self._loaded_variant = None

        # 尝试释放显存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("已清空 CUDA 缓存")
        except ImportError:
            pass

    # ===================================================================
    # generate
    # ===================================================================

    def generate(
        self,
        text: str,
        mode: str = "clone",
        reference_audio_path: Optional[str] = None,
        emotion_audio_path: Optional[str] = None,
        emotion_alpha: float = 1.0,
        emotion_vector: Optional[list[float]] = None,
        emotion_text: str = "",
        do_sample: bool = True,
        top_p: float = 0.8,
        top_k: int = 30,
        temperature: float = 0.8,
        repetition_penalty: float = 10.0,
        max_mel_tokens: int = 1500,
        max_text_tokens_per_segment: int = 120,
        use_random: bool = False,
        output_path: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        """同步生成音频。

        IndexTTS2 是零样本克隆模型，**必须有参考音频**（spk_audio_prompt）。
        ``infer()`` 写文件（output_path），不返回 numpy 数组，需用
        ``soundfile.read()`` 读回。

        5 种模式映射（v0.1 仅前 3 种，emo_text 留 v0.2）：
            - ``clone``      : 音色参考音频 + 文本
            - ``emo_clone``  : 音色参考 + 情感参考音频 + α
            - ``emo_vector`` : 音色参考 + 8 维情感向量
            - ``emo_text``   : 音色参考 + 情感描述文本（v0.2 启用，依赖 Qwen3）

        Args:
            text: 要合成的文本
            mode: 合成模式（clone/emo_clone/emo_vector/emo_text）
            reference_audio_path: 音色参考音频路径（必填，零样本克隆需要）
            emotion_audio_path: 情感参考音频路径（emo_clone 模式）
            emotion_alpha: 情感权重 0.0-1.0（emo_clone 模式）
            emotion_vector: 8 维情感向量（emo_vector 模式）
            emotion_text: 情感描述文本（emo_text 模式，v0.2 启用）
            do_sample: 是否采样
            top_p: nucleus sampling
            top_k: top-k sampling
            temperature: 采样温度
            repetition_penalty: 重复惩罚
            max_mel_tokens: 最大 mel token 数
            max_text_tokens_per_segment: 单段最大文本 token
            use_random: 随机采样
            output_path: 输出 wav 文件路径（None 时自动生成临时文件）

        Returns:
            (samples, sample_rate): 音频采样数据和采样率

        Raises:
            RuntimeError: 模型未加载或生成失败
            ValueError: 参数缺失或模式不支持
        """
        with self._lock:
            if self._model is None:
                raise RuntimeError("模型未加载，请先调用 load()")

            # IndexTTS2 是零样本克隆模型，必须有参考音频
            if not reference_audio_path:
                raise ValueError(
                    "IndexTTS2 必须提供参考音频（零样本克隆模型）。"
                    "请通过 audios[role=speaker] 或 inputs.reference_audio_path 传入。"
                )

            # 构造 generation_kwargs（对齐 webui.py 预设）
            kwargs: dict[str, Any] = {
                "verbose": False,
                "max_text_tokens_per_segment": max_text_tokens_per_segment,
                "do_sample": do_sample,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
                "max_mel_tokens": max_mel_tokens,
                "use_random": use_random,
            }

            # 模式映射 → IndexTTS2.infer() 参数
            spk_audio_prompt = reference_audio_path

            if mode == "clone":
                # 基础克隆：音色参考音频 + 文本
                pass  # 无额外参数

            elif mode == "emo_clone":
                # 情感克隆：音色参考 + 情感参考音频 + α
                if not emotion_audio_path:
                    raise ValueError("emo_clone 模式需要 emotion_audio_path")
                kwargs["emo_audio_prompt"] = emotion_audio_path
                kwargs["emo_alpha"] = emotion_alpha

            elif mode == "emo_vector":
                # 情感向量：音色参考 + 8 维情感向量
                if not emotion_vector or len(emotion_vector) != 8:
                    raise ValueError(
                        "emo_vector 模式需要 8 维情感向量 "
                        "[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]"
                    )
                kwargs["emo_vector"] = emotion_vector

            elif mode == "emo_text":
                # 情感文本：音色参考 + 情感描述文本（v0.2 启用，依赖 Qwen3）
                if not emotion_text:
                    raise ValueError("emo_text 模式需要 emotion_text")
                kwargs["use_emo_text"] = True
                kwargs["emo_text"] = emotion_text

            else:
                raise ValueError(
                    f"未知合成模式: {mode}（支持: clone/emo_clone/emo_vector/emo_text）"
                )

            # 临时输出路径（IndexTTS2.infer() 写文件）
            temp_file = None
            if not output_path:
                temp_file = tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False, prefix="indextts2_"
                )
                output_path = temp_file.name
                temp_file.close()

            logger.info(
                "IndexTTS2 生成: mode=%s, text=%r, ref=%s, output=%s, "
                "temperature=%.2f, top_p=%.2f, top_k=%d",
                mode,
                text[:50] + ("..." if len(text) > 50 else ""),
                reference_audio_path,
                output_path,
                temperature, top_p, top_k,
            )

            # 调用推理（写文件）
            try:
                self._model.infer(
                    spk_audio_prompt=spk_audio_prompt,
                    text=text,
                    output_path=output_path,
                    **kwargs,
                )
            except Exception as e:
                logger.error("IndexTTS2 生成失败: %s", e)
                # 清理临时文件
                if temp_file:
                    try:
                        Path(output_path).unlink(missing_ok=True)
                    except Exception:
                        pass
                raise RuntimeError(f"IndexTTS2 生成失败: {e}") from e

            # 读回 numpy 数组（IndexTTS2.infer() 写文件，不返回数组）
            try:
                import soundfile as sf
                samples, sr = sf.read(output_path)
                # 确保是 float32 且 1 维（单声道）
                if samples.dtype != np.float32:
                    samples = samples.astype(np.float32)
                if samples.ndim > 1:
                    samples = samples[:, 0]  # 取左声道
            except Exception as e:
                logger.error("读取生成音频文件失败: %s, path=%s", e, output_path)
                raise RuntimeError(f"读取生成音频文件失败: {e}") from e
            finally:
                # 清理临时文件
                if temp_file:
                    try:
                        Path(output_path).unlink(missing_ok=True)
                    except Exception:
                        pass

            logger.info(
                "IndexTTS2 生成完成: samples_shape=%s, duration=%.2fs, sr=%d",
                samples.shape,
                len(samples) / sr,
                sr,
            )
            return samples, sr

    # ===================================================================
    # 文件完整性校验（project-rules.md §第二部分 §2）
    # ===================================================================

    def _check_files_integrity(self, variant_path: str) -> None:
        """校验主模型 + 辅助模型文件完整性（不下载，只校验）。

        校验逻辑：
            - 文件存在（``exists()``）
            - 文件大小 >= 阈值（防止残文件 / 断点续传未完成）
            - 缺失时抛 ``RuntimeError``，包含完整下载指引

        Args:
            variant_path: 模型目录路径

        Raises:
            RuntimeError: 文件缺失或大小不足，包含缺失列表 + 本地路径 +
                         HF 仓库 ID + "请到模型管理界面下载"指引
        """
        base = Path(variant_path)
        hf_cache = base / "hf_cache"
        qwen_emo_dir = base / "qwen0.6bemo4-merge"
        missing: list[str] = []
        incomplete: list[str] = []

        # ----- 1. 主模型文件（7 个） -----
        for filename, min_size in _MAIN_MODEL_FILES:
            filepath = base / filename
            if not filepath.exists():
                missing.append(f"主模型/{filename}")
            elif filepath.stat().st_size < min_size:
                incomplete.append(
                    f"主模型/{filename} (size={filepath.stat().st_size} < {min_size})"
                )

        # ----- 2. Qwen 情感分析子模型（11 个文件，~1.2GB） -----
        # v0.2.16+ 新增：IndexTTS2 库通过 QwenEmotion 加载 config.yaml 的
        # qwen_emo_path: qwen0.6bemo4-merge/ 子目录下的 Qwen 模型。
        for rel_path, min_size in _QWEN_EMO_FILES:
            filepath = qwen_emo_dir / rel_path
            if not filepath.exists():
                missing.append(f"Qwen情感模型/{rel_path}")
            elif filepath.stat().st_size < min_size:
                incomplete.append(
                    f"Qwen情感模型/{rel_path} "
                    f"(size={filepath.stat().st_size} < {min_size})"
                )

        # ----- 3. 辅助模型文件（4 个独立 HF 仓库） -----
        for aux_name, files in _AUX_MODEL_FILES.items():
            for rel_path, min_size in files:
                filepath = hf_cache / rel_path
                if not filepath.exists():
                    missing.append(f"辅助模型[{aux_name}]/{rel_path}")
                elif filepath.stat().st_size < min_size:
                    incomplete.append(
                        f"辅助模型[{aux_name}]/{rel_path} "
                        f"(size={filepath.stat().st_size} < {min_size})"
                    )

        # ----- 4. 缺失或不完整时抛 RuntimeError -----
        if missing or incomplete:
            # 构造完整错误信息（project-rules.md §第二部分 §2 代码模板）
            repo_ids = [
                "IndexTeam/IndexTTS-2（主模型 + qwen0.6bemo4-merge/ Qwen 情感分析子模型）",
                "facebook/w2v-bert-2.0（辅助模型：w2v-bert 特征提取器）",
                "amphion/MaskGCT（辅助模型：semantic_codec 语义编码器）",
                "funasr/campplus（辅助模型：CAMPPlus 说话人编码器）",
                "nvidia/bigvgan_v2_22khz_80band_256x（辅助模型：BigVGAN 声码器）",
            ]

            parts = ["模型文件不完整："]
            if missing:
                parts.append(f"\n  缺失 {len(missing)} 个文件：")
                for m in missing:
                    parts.append(f"    - {m}")
            if incomplete:
                parts.append(f"\n  不完整（大小不足）{len(incomplete)} 个文件：")
                for i in incomplete:
                    parts.append(f"    - {i}")

            parts.append(f"\n\n本地路径：{base}")
            parts.append("\nHuggingFace 仓库：")
            for r in repo_ids:
                parts.append(f"  - {r}")
            parts.append(
                "\n\n请到引擎详情页 → 模型管理 → 点击「下载缺失文件」按钮，"
                "选择镜像源后下载。"
            )

            raise RuntimeError("".join(parts))

        # 5. 残文件扫描（v0.3.5+ 新增，防止断点续传未完成文件干扰加载）
        # 残文件（.incomplete/.tmp/.partial）可能被 indextts 库内部的旧版字段名
        # fallback 加载逻辑误读（如 emo_matrix.pt.incomplete 被当作 emo_matrix.pt），
        # 导致 0 字节空文件加载失败 → 抛异常 → wrapper.load() 返回 500。
        # 设计原则：不阻断加载（残文件存在不代表真正加载会失败），只记 WARNING。
        # 真正失败时由 wrapper.load() 的 except 块 + 已修复的 Bug-B 输出完整 Traceback。
        _RESIDUAL_PATTERNS = ("*.incomplete", "*.tmp", "*.partial")
        residuals: list[str] = []
        for pattern in _RESIDUAL_PATTERNS:
            # 扫描主模型目录
            residuals.extend(str(p) for p in base.glob(pattern))
            # 扫描 hf_cache 子目录（辅助模型整仓下载可能残留）
            residuals.extend(str(p) for p in hf_cache.rglob(pattern))
            # 扫描 qwen_emo 子目录（Qwen 模型整仓下载可能残留）
            residuals.extend(str(p) for p in qwen_emo_dir.rglob(pattern))

        if residuals:
            logger.warning(
                "发现 %d 个残文件（可能干扰加载，建议清理）: %s",
                len(residuals), residuals,
            )
            # 不抛错，只在错误信息中提示
            # 实际加载失败时由 wrapper.load() 的 except 块携带此信息
        else:
            logger.debug("残文件扫描通过：未发现 .incomplete/.tmp/.partial 文件")

        logger.info(
            "文件完整性校验通过: variant_path=%s, 主模型=%d 文件, "
            "Qwen情感模型=%d 文件, 辅助模型=%d 仓库",
            base, len(_MAIN_MODEL_FILES), len(_QWEN_EMO_FILES), len(_AUX_MODEL_FILES),
        )

    # ===================================================================
    # aux_paths 构造
    # ===================================================================

    def _resolve_aux_paths(self, variant_path: str) -> dict[str, str]:
        """构造 aux_paths 字典，指向预下载的辅助模型路径。

        IndexTTS2 构造函数的 ``aux_paths`` 参数用于绕过
        ``ensure_models_available()`` 自动下载。返回的字典结构与
        ``indextts.utils.model_download.ensure_models_available()`` 一致：

        .. code-block:: python

            {
                "w2v_bert": "{variant_path}/hf_cache/w2v-bert-2.0/",   # 目录
                "semantic_codec": "{variant_path}/hf_cache/semantic_codec_model.safetensors",  # 文件
                "campplus": "{variant_path}/hf_cache/campplus_cn_common.bin",  # 文件
                "bigvgan": "{variant_path}/hf_cache/bigvgan/",         # 目录
            }

        Args:
            variant_path: 模型目录路径

        Returns:
            aux_paths 字典
        """
        hf_cache = Path(variant_path) / "hf_cache"
        return {
            "w2v_bert": str(hf_cache / "w2v-bert-2.0"),
            "semantic_codec": str(hf_cache / "semantic_codec_model.safetensors"),
            "campplus": str(hf_cache / "campplus_cn_common.bin"),
            "bigvgan": str(hf_cache / "bigvgan"),
        }

    # ===================================================================
    # 采样率读取
    # ===================================================================

    def _read_sample_rate(self, variant_path: str) -> int:
        """从 config.yaml 读取实际采样率。

        IndexTTS2 的采样率由 ``config.yaml`` 的
        ``s2mel.preprocess_params.sr`` 字段决定（默认 24000）。

        Args:
            variant_path: 模型目录路径

        Returns:
            采样率（Hz），读取失败时返回默认值 24000
        """
        try:
            import yaml
            config_path = Path(variant_path) / "config.yaml"
            if not config_path.exists():
                logger.warning("config.yaml 不存在，使用默认采样率 %d", _DEFAULT_SAMPLE_RATE)
                return _DEFAULT_SAMPLE_RATE

            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

            # 尝试从 s2mel.preprocess_params.sr 读取
            s2mel = cfg.get("s2mel", {}) if isinstance(cfg, dict) else {}
            preprocess = s2mel.get("preprocess_params", {}) if isinstance(s2mel, dict) else {}
            sr = preprocess.get("sr", _DEFAULT_SAMPLE_RATE)
            sr = int(sr)
            logger.info("从 config.yaml 读取采样率: %d", sr)
            return sr
        except Exception as e:
            logger.warning("读取 config.yaml 采样率失败，使用默认值 %d: %s", _DEFAULT_SAMPLE_RATE, e)
            return _DEFAULT_SAMPLE_RATE

    # ===================================================================
    # 设备解析 / 显存预检
    # ===================================================================

    def _resolve_device(self, device: str) -> str:
        """解析设备字符串（project-rules.md §第二部分 §4）。

        Args:
            device: auto / cpu / cuda:0 / cuda:1

        Returns:
            实际设备字符串（cuda:0 / cpu）
        """
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda:0"
                return "cpu"
            except ImportError:
                return "cpu"
        return device

    def _get_device_name(self, device: str) -> str:
        """获取设备的显示名称。

        Args:
            device: cuda:0 / cpu / cuda:1 等

        Returns:
            设备显示名称：
                - cuda:0 → "NVIDIA GeForce RTX 4090"（实际 GPU 名称）
                - cpu    → "CPU"
        """
        if device == "cpu":
            return "CPU"
        try:
            import torch
            if torch.cuda.is_available():
                gpu_idx = int(device.split(":")[1]) if ":" in device else 0
                name = torch.cuda.get_device_name(gpu_idx)
                return name or device
        except (ImportError, ValueError, RuntimeError):
            pass
        return device

    def _check_vram_available(self, device: str, required_gb: float = _REQUIRED_VRAM_GB_FP16) -> bool:
        """加载前预检显存（详见 architecture.md §3.5 显存预检机制）。

        空闲显存 < required_gb 时的行为（v0.3.5+ 按设备策略区分）：
            - gpu+cpu 策略：记录警告并返回 False（调用方降级 CPU 加载）
            - 其余策略（auto/gpu）：抛 RuntimeError（中文提示 + 解决建议），
              不尝试加载，避免 PyTorch OOM 异常的原始英文信息暴露给用户。

        Args:
            device: 推理设备
            required_gb: 模型预估需要的显存（GB），IndexTTS2 FP16 ≈ 6GB

        Returns:
            bool: True=预检通过（或 CPU 模式/检测异常跳过）；False=显存不足
                  （仅 gpu+cpu 策略会返回 False，其余策略显存不足直接抛错）

        Raises:
            RuntimeError: 显存不足（auto/gpu 策略）
        """
        if not device.startswith("cuda"):
            return True  # CPU 模式无需预检

        try:
            import torch
            if not torch.cuda.is_available():
                return True
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024 ** 3)
            if free_gb < required_gb:
                if self._device_policy == "gpu+cpu":
                    logger.warning(
                        "显存预检不足: free=%.2fGB < required=%.2fGB（gpu+cpu 策略，降级 CPU）",
                        free_gb, required_gb,
                    )
                    return False
                logger.warning(
                    "显存预检失败: free=%.2fGB < required=%.2fGB",
                    free_gb, required_gb,
                )
                raise RuntimeError(
                    f"GPU 显存不足：空闲 {free_gb:.2f}GB < 需要 {required_gb}GB。"
                    f"建议：1) 关闭其他占用显存的程序 2) 切换到 CPU 模式 3) 卸载其他已加载的引擎"
                )
            logger.info("显存预检通过: free=%.2fGB >= required=%.2fGB", free_gb, required_gb)
            return True
        except RuntimeError as e:
            if "显存不足" in str(e):
                raise  # 友好错误直接抛出
            logger.warning("显存预检异常(跳过): %s", e)
        except Exception as e:
            logger.warning("显存预检异常(跳过): %s", e)
        return True

    def get_vram_used(self) -> int:
        """获取已用显存（MB）。"""
        try:
            import torch
            if torch.cuda.is_available() and "cuda" in self._device:
                return int(torch.cuda.memory_allocated() / 1024 / 1024)
        except ImportError:
            pass
        return 0
