"""IndexTTS2 TTS 引擎推理核心（v0.4.3+ 引入）。

继承 ``CustomEngine``（全自定义模式 / integrated 范式），实现 7 个
``_xxx_impl`` 方法，封装 IndexTTS2Wrapper 的加载/推理/卸载流程。

8 端点契约（由 SDK BaseEngine 提供，本文件只实现 _impl）：
    - ``/capabilities``    返回扫描到的 IndexTTS2 模型变体
    - ``/load``            调用 IndexTTS2Wrapper.load() 加载模型权重
    - ``/unload``          调用 IndexTTS2Wrapper.unload() 释放显存
    - ``/infer``           解析 IndexTTS2Inputs → 调用 generate() → 保存音频 → SSE 推送
    - ``/status``          返回引擎状态（已加载变体 / 设备 / 显存）
    - ``/lora/load``       v0.1 不支持，返回 501
    - ``/lora/unload``     v0.1 不支持，返回 501

音色库集成（v0.4.3+ 方案 A）：
    - 完全使用平台音色库（``/v1/voices``），废弃 IndexTTS2 原生 presets
    - gateway 把 ``extra.voice_id`` 解析为 ``audios[role=speaker].path``
    - 引擎只需从 ``request.audios`` 提取，无需自己查音色库
    - 详见 PR/development-guide.md §28.25

规范：PR/development-guide.md §28.20A / project-rules.md §第二部分
参考：``engines/voxcpm_tts_c4a8f2e7/engine.py``
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from aireliable_engine import (
    CapabilitiesResponse,
    CustomEngine,
    InferRequest,
    LoadResponse,
    LoraLoadResponse,
    LoraUnloadResponse,
    SSEStream,
    StatusResponse,
    UnloadResponse,
    VariantInfo,
)

from .audio_utils import get_audio_duration, save_audio
from .config import EngineConfig
from .indextts2_wrapper import IndexTTS2Wrapper
from .model_scanner import ModelScanner
from .progress_reporter import ProgressReporter
from .schemas import EMOTION_LABELS, INDEXTTS2_MODES, IndexTTS2Inputs

logger = logging.getLogger(__name__)


# ===========================================================================
# 能力声明扩展
# ===========================================================================

class IndexTTS2VariantInfo(VariantInfo):
    """IndexTTS2 变体能力声明（扩展 SDK VariantInfo）。

    在标准字段之外补充 IndexTTS2 特有信息：
    采样率 / 支持语言 / 合成模式 / 8 维情感标签。

    Pydantic ``model_dump()`` 会序列化全部字段，前端可读取这些扩展字段
    渲染 IndexTTS2 专属 UI（如情感向量滑块 / 模式选择器）。
    """

    model_config = ConfigDict(extra="allow")

    sample_rate: int = Field(24000, description="输出采样率（Hz）")
    languages: list[str] = Field(default_factory=lambda: ["zh", "en"])
    modes: list[str] = Field(
        default_factory=lambda: list(INDEXTTS2_MODES[:3]),
        description="支持模式（v0.1 仅 clone/emo_clone/emo_vector）",
    )
    emotion_labels: list[str] = Field(
        default_factory=lambda: list(EMOTION_LABELS),
        description="8 维情感标签",
    )


# ===========================================================================
# 引擎主体
# ===========================================================================

class IndexTTS2Engine(CustomEngine):
    """IndexTTS2 TTS 引擎。

    全自定义模式（integrated 范式），直接控制模型加载/推理流程，
    不走工作流节点系统。UI 挂载 Gradio 5.45.0（由 server.py 处理）。

    与 VoxCPMEngine 的关键差异：
        - IndexTTS2 是零样本克隆模型，**必须有参考音频**（spk_audio_prompt）
        - 支持 4 种情感控制模式（clone/emo_clone/emo_vector/emo_text）
        - ``infer()`` 写文件，不返回 numpy 数组（wrapper 内部读回）
        - 采样率 24kHz（VoxCPM 为 48kHz/16kHz）
    """

    def __init__(self, config: EngineConfig):
        super().__init__(
            engine_id=config.engine_id,
            engine_name=config.engine_name,
            capability=config.capability,
            engine_type=config.engine_type,
        )
        self.config: EngineConfig = config

        # 核心组件
        self.wrapper = IndexTTS2Wrapper(config)
        # v0.2.11+ 方案 B：注入 engine=self，启用平台 API 声明式检测
        # （平台不可达时 fallback 到本地扫描）
        self.scanner = ModelScanner(config.model_dir, engine=self)

        # 启动时扫描模型目录（同步初始化为空，异步扫描在 lifespan 中触发）
        self._available_variants: list = []
        # __init__ 阶段平台 API 可能尚未就绪，先用 fallback 本地扫描
        self._refresh_scan()

        # v0.1 不支持 LoRA
        self._available_loras: list[str] = []

        # 输出根目录（供 BaseEngine.get_output_path 使用）
        self.output_root = config.output_root

        # 读取 manifest 中的 default_variant（供 BaseEngine.infer auto-load 使用）
        # 优先级：request.variant > self.default_variant > 第一个 available variant
        self.default_variant = self._read_default_variant_from_manifest(
            Path(__file__).parent / "manifest.yaml"
        )

    # ===================================================================
    # 扫描
    # ===================================================================

    def _refresh_scan(self) -> None:
        """刷新模型扫描结果（同步 fallback，用于 __init__ 阶段）。

        v0.2.11+ 方案 B：此方法走 fallback 本地扫描逻辑。
        平台 API 异步扫描请调用 ``_refresh_scan_async()``（lifespan startup 阶段调用）。
        """
        logger.info("刷新模型扫描: %s", self.config.model_dir)
        self._available_variants = self.scanner.scan_variants()

    async def _refresh_scan_async(self) -> None:
        """v0.2.11+ 方案 B：异步刷新模型扫描结果（优先平台 API）。

        在 lifespan startup 阶段调用，此时平台 API 已就绪。
        平台不可达时 fallback 到本地扫描。
        """
        logger.info("异步刷新模型扫描（方案 B 平台 API）: %s", self.config.model_dir)
        try:
            self._available_variants = await self.scanner.scan_variants_async()
        except Exception as e:
            logger.warning("异步扫描失败，回退到本地扫描: %s", e)
            self._available_variants = self.scanner.scan_variants()

    # ===================================================================
    # 7 个 _xxx_impl 实现
    # ===================================================================

    async def _capabilities_impl(self) -> CapabilitiesResponse:
        """GET /capabilities：返回可用模型变体。"""
        variants: list[VariantInfo] = []
        for v in self._available_variants:
            # vram_required 在 IndexTTS2Capability 中是 int（MB），转成描述字符串
            vram_mb = getattr(v, "vram_required", 0)
            vram_str = f"{vram_mb}MB" if vram_mb else "未知"

            variants.append(IndexTTS2VariantInfo(
                name=v.name,
                base_model=v.name,                  # IndexTTS2 无 base_model 概念，用 name
                precision=v.precision,
                loader="indextts2",
                vram_required=vram_str,
                available=True,
                sample_rate=v.sample_rate,
                languages=v.languages,
                modes=v.modes,
                emotion_labels=v.emotion_labels,
            ))

        return CapabilitiesResponse(
            engine_id=self.engine_id,
            engine_name=self.engine_name,
            capability=self.capability,
            type=self.engine_type,
            variants=variants,
            loras=self._available_loras,
            supported_sizes=[],                      # TTS 不用 size 概念
            default_variant=self.default_variant,
            # v0.4.1+ MCP 化：推理输入 JSON Schema（MCP 专属层动态 tool 数据源）
            input_schema=IndexTTS2Inputs.model_json_schema(),
        )

    async def _load_impl(self, variant: str) -> LoadResponse:
        """POST /load：加载 IndexTTS2 模型权重。

        ``IndexTTS2.__init__()`` 是同步阻塞调用（内部加载多个子模型），
        用 ``asyncio.to_thread`` 包装避免阻塞事件循环。

        v0.2.15+ 失败时记录最近日志行，便于前端 ``recent_logs`` 显示。
        """
        # variant=None 回退链（与 FLUX 引擎范式一致）：
        # WebUI「自动（默认变体）」选项 / gateway 未指定时 variant 为 None
        # 优先级：request.variant > self.default_variant > 第一个 available variant
        if variant is None:
            variant = self.default_variant or (
                self._available_variants[0].name if self._available_variants else None
            )
            logger.info("load()：variant 未指定，回退到: %s", variant)
        if variant is None:
            raise FileNotFoundError(
                "无可加载的 variant（未指定且无默认/可用变体），"
                "请到模型管理界面下载模型"
            )
        variant_path = self.scanner.resolve_variant_path(variant)
        if variant_path is None:
            raise FileNotFoundError(
                f"variant 不存在: {variant} "
                f"（可用: {[v.name for v in self._available_variants]}）"
            )

        logger.info("开始加载 variant: %s (path=%s)", variant, variant_path)

        start_time = time.time()
        try:
            await asyncio.to_thread(
                self.wrapper.load,
                variant,              # variant 名（存入 _loaded_variant）
                str(variant_path),    # variant 路径（用于加载权重）
                self.config.device,
            )
        except Exception as e:
            logger.exception("IndexTTS2 模型加载失败: variant=%s", variant)
            raise RuntimeError(f"IndexTTS2 模型加载失败: {e}") from e

        load_time = time.time() - start_time
        vram_used = self.wrapper.get_vram_used()

        logger.info(
            "加载完成: variant=%s, device=%s, sample_rate=%d, load_time=%.2fs",
            variant, self.wrapper.device, self.wrapper.sample_rate, load_time,
        )

        return LoadResponse(
            variant=variant,
            state="ready",
            vram_used=f"{vram_used}MB",
            load_time=load_time,
        )

    async def _unload_impl(self) -> UnloadResponse:
        """POST /unload：卸载模型，释放显存。"""
        if not self.wrapper.is_loaded:
            return UnloadResponse(state="idle", vram_freed="0MB")

        vram_freed = self.wrapper.get_vram_used()
        await asyncio.to_thread(self.wrapper.unload)

        logger.info("卸载完成: vram_freed=%dMB", vram_freed)
        return UnloadResponse(
            state="idle",
            vram_freed=f"{vram_freed}MB",
        )

    async def _infer_impl(self, request: InferRequest, stream: SSEStream) -> None:
        """POST /infer：TTS 推理（SSE 流式推送进度）。

        流程：
            1. 解析 IndexTTS2Inputs（从 request.inputs）
            2. 提取音频输入（从 request.audios，gateway 已解析 voice_id）
               - ``audios[role=speaker]`` → reference_audio_path（音色参考）
               - ``audios[role=emotion]`` → emotion_audio_path（情感参考）
            3. 模式决策：有 emotion 音频 → emo_clone / 否则按 inputs.mode
            4. 上报 start / model_ready / text_processing 帧
            5. 线程池执行 wrapper.generate()（同步阻塞，带心跳）
            6. 保存音频到 output/audio/{task_id}.wav
            7. 上报 completed 帧（含 audio_url）

        错误处理：内部捕获异常，推送 error 帧后 return，
        不重新抛出（避免 BaseEngine.infer 把状态置为 ERROR，
        推理失败后模型仍可复用）。

        v0.3.4+ 集成 SDK 任务上报：
            - 入口生成 task_id（若 request.task_id 为 None）
            - 调 reporter.report_created 上报到 gateway（status=queued）
            - emit_start 自动触发 fire_report_running
            - emit_completed/emit_error 自动触发 fire_report_succeeded/failed

        音色库集成（v0.4.3+ 方案 A）：
            gateway 把 ``extra.voice_id`` 解析为 ``audios[role=speaker].path``，
            引擎只需从 request.audios 提取，无需自己查音色库。
        """
        # ----- 0. 生成 task_id（v0.3.4+ T7）-----
        task_id = request.task_id or f"tts_{uuid.uuid4().hex[:16]}"
        reporter = ProgressReporter(stream, task_id=task_id)

        # ----- 1. 解析输入 -----
        try:
            inputs = IndexTTS2Inputs(**request.inputs)
        except Exception as e:
            logger.error("IndexTTS2Inputs 解析失败: %s", e)
            reporter.emit_error(f"输入参数无效: {e}", "invalid_inputs")
            return

        # ----- 2. 提取音频输入（gateway 已把 voice_id 解析为 audios[role=speaker].path）-----
        reference_audio_path = inputs.reference_audio_path
        emotion_audio_path = inputs.emotion_audio_path
        for audio in request.audios:
            if not audio.path:
                continue
            if audio.role == "speaker":
                reference_audio_path = audio.path
            elif audio.role == "emotion":
                emotion_audio_path = audio.path

        # ----- 2.5 模式决策：有 emotion 音频 → emo_clone（覆盖 inputs.mode）-----
        # v0.4.3+ 方案 B+C 混合：音色走 voice_id，情感走 audios[role=emotion] 直传
        # 详见 PR/development-guide.md §28.25
        mode = inputs.mode
        if emotion_audio_path and mode == "clone":
            mode = "emo_clone"
            logger.info("检测到 audios[role=emotion]，模式自动切换 clone → emo_clone")
        # 兼容：gateway 同时写入 workflow 和 mode
        if inputs.workflow and not inputs.mode:
            mode = inputs.workflow

        # ----- 3. v0.3.4+ 上报任务创建到 gateway（status=queued）-----
        await reporter.init_task_reporter()
        await reporter.report_created(
            request_summary={
                "text": inputs.text[:200] if inputs.text else "",
                "mode": mode,
            },
            variant=request.variant or "",
        )

        # ----- 4. 上报开始阶段 -----
        reporter.emit_start(inputs.text, mode)
        reporter.emit_model_ready(
            self.wrapper.loaded_variant or "unknown",
            self.wrapper.device,
        )

        # ----- 5. 模式参数校验 -----
        # IndexTTS2 是零样本克隆模型，必须有参考音频
        if not reference_audio_path:
            reporter.emit_error(
                "IndexTTS2 必须提供参考音频（零样本克隆模型）。"
                "请通过 audios[role=speaker] 或 inputs.reference_audio_path 传入，"
                "或通过 extra.voice_id 引用平台音色库。",
                "missing_reference_audio",
            )
            return

        if mode == "emo_clone" and not emotion_audio_path:
            reporter.emit_error(
                "emo_clone 模式需要情感参考音频（通过 audios[role=emotion] 传入）",
                "missing_emotion_audio",
            )
            return

        if mode == "emo_vector" and (not inputs.emotion_vector or len(inputs.emotion_vector) != 8):
            reporter.emit_error(
                "emo_vector 模式需要 8 维情感向量 "
                "[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]",
                "missing_emotion_vector",
            )
            return

        reporter.emit_text_processing(inputs.text)

        # ----- 6. 线程池执行推理（带心跳，重置 gateway httpx read timeout）-----
        def _generate() -> tuple:
            return self.wrapper.generate(
                text=inputs.text,
                mode=mode,
                reference_audio_path=reference_audio_path,
                emotion_audio_path=emotion_audio_path,
                emotion_alpha=inputs.emotion_alpha,
                emotion_vector=inputs.emotion_vector,
                emotion_text=inputs.emotion_text or "",
                do_sample=inputs.do_sample,
                top_p=inputs.top_p,
                top_k=inputs.top_k,
                temperature=inputs.temperature,
                repetition_penalty=inputs.repetition_penalty,
                max_mel_tokens=inputs.max_mel_tokens,
                max_text_tokens_per_segment=inputs.max_text_tokens_per_segment,
            )

        # 心跳任务：generate() 是同步阻塞调用，期间无 SSE 帧输出，
        # 会导致 gateway httpx read timeout 触发 → 502 Bad Gateway。
        # 在 to_thread 执行期间定期推送进度帧，重置 gateway 的 read timeout 计时。
        # 心跳间隔 30s，远小于 gateway 超时（默认 1800s），双重保险。
        async def _heartbeat():
            elapsed = 0
            while True:
                await asyncio.sleep(30)
                elapsed += 30
                # 进度从 0.15 渐进到 0.80，避免超过推理完成后的 0.85
                progress = min(0.80, 0.15 + elapsed * 0.005)
                reporter.emit_generating(progress, elapsed=elapsed)

        heartbeat_task = asyncio.create_task(_heartbeat())
        try:
            samples, sample_rate = await asyncio.to_thread(_generate)
        except Exception as e:
            logger.exception("IndexTTS2 推理失败")
            reporter.emit_error(f"推理失败: {e}", "infer_failed")
            return
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

        reporter.emit_generating(0.85, samples_count=len(samples))

        # ----- 7. 保存音频 -----
        reporter.emit_saving()

        output_filename = f"{task_id}.{inputs.output_format}"
        output_path = self.get_output_path("audio", output_filename)

        try:
            saved_path = await asyncio.to_thread(
                save_audio,
                samples,
                sample_rate,
                output_path,
                inputs.output_format,
            )
        except Exception as e:
            logger.exception("保存音频失败")
            reporter.emit_error(f"保存音频失败: {e}", "save_failed")
            return

        # ----- 8. 上报完成 -----
        # audio_url 为相对路径（server.py 把 output_root 挂载到 /output）
        audio_url = f"/output/audio/{saved_path.name}"
        # audio_path 为本地绝对路径（用于 Gradio UI 播放，绕过 SSRF 防护）
        audio_path = str(saved_path.resolve())
        duration = get_audio_duration(samples, sample_rate)

        logger.info(
            "IndexTTS2 推理完成: task_id=%s, mode=%s, duration=%.2fs, "
            "audio_url=%s, audio_path=%s",
            task_id, mode, duration, audio_url, audio_path,
        )

        await reporter.emit_completed(
            audio_url=audio_url,
            duration=duration,
            sample_rate=sample_rate,
            mode=mode,
            audio_path=audio_path,
        )

    async def _status_impl(self) -> StatusResponse:
        """GET /status：返回引擎状态。"""
        vram_used = self.wrapper.get_vram_used()
        return StatusResponse(
            state=self.state_machine.state.value,
            loaded_variant=self.wrapper.loaded_variant,
            vram_used=f"{vram_used}MB",
            loaded_loras=[],                      # v0.1 不支持 LoRA
            last_used_at=self.state_machine.last_used_at,
            uptime=self.state_machine.uptime,
            extra={
                "device": self.wrapper.device,
                "device_name": self.wrapper.device_name,
                "sample_rate": self.wrapper.sample_rate,
                "is_loaded": self.wrapper.is_loaded,
            },
        )

    async def _load_lora_impl(self, name: str, weight: float) -> LoraLoadResponse:
        """POST /lora/load：v0.1 不支持，返回 501。"""
        from fastapi import HTTPException
        raise HTTPException(
            status_code=501,
            detail="IndexTTS2 TTS 引擎 v0.1 不支持运行时 LoRA 切换",
        )

    async def _unload_lora_impl(self, name: str) -> LoraUnloadResponse:
        """POST /lora/unload：v0.1 不支持，返回 501。"""
        from fastapi import HTTPException
        raise HTTPException(
            status_code=501,
            detail="IndexTTS2 TTS 引擎 v0.1 不支持运行时 LoRA 切换",
        )

    # ===================================================================
    # 辅助方法
    # ===================================================================

    def _read_default_variant_from_manifest(self, manifest_path: Path) -> Optional[str]:
        """从 manifest.yaml 读取 default_variant 字段。

        BaseEngine.infer 的 auto-load 优先级：
            request.variant > self.default_variant > 第一个 available variant

        Args:
            manifest_path: manifest.yaml 路径

        Returns:
            default_variant 名称（如 "IndexTeam/IndexTTS-2"），读取失败返回 None
        """
        try:
            import yaml
            if not manifest_path.exists():
                logger.warning("manifest.yaml 不存在: %s", manifest_path)
                return None
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            models = data.get("models", {}) if isinstance(data, dict) else {}
            default = models.get("default_variant")
            if default:
                logger.info("从 manifest.yaml 读取 default_variant: %s", default)
            return default
        except Exception as e:
            logger.warning("读取 manifest.yaml default_variant 失败: %s", e)
            return None
