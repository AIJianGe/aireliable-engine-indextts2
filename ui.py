"""IndexTTS2 TTS 引擎 Gradio Web UI（Gradio 5.45.0）。

v0.4.4+ 单 Tab 化设计：
    - 1 个 Tab：情感控制（覆盖原 基础 TTS + 克隆 + 情感控制 三个 Tab）
    - 顶部：平台音色库下拉（仅加载音色，不暴露管理）+ 🔄 刷新
    - 音色参考：必填（从音色库选 或 直接上传）
    - 合成模式：Radio 4 选 1（clone / emo_clone / emo_vector / emo_text）
        根据选择动态展开对应参数区（gr.Group visible）
    - 高级参数：Accordion 折叠（top_p / top_k / temperature / ...）
    - 合成文本：Textbox
    - 生成 + 合成结果 Audio

设计规范（PR/development-guide.md §28.20A.5 / §28.25）：
    - UI 通过 httpx 调自身 API（``engine_url``），**音色库走 SDK ``VoiceLibraryClient``**
    - 不能直接调 wrapper（平台规范）
    - v0.3.5+：音色库调用从裸 httpx 改为 SDK ``VoiceLibraryClient``（sync 接口），
      与 ``PlatformTaskReporter`` / ``PlatformVramManager`` 模式一致
    - 音色库走 gateway ``/v1/voices*``，废弃 IndexTTS2 原生 presets
    - 跨引擎复用：VoxCPM 创建的音色在 IndexTTS2 中等价使用
    - v0.4.4+：音色管理操作（上传/删除/试听）由平台音色库管理页负责，
      引擎 UI 不再重复暴露

Gradio 版本说明（INDEXTTS2_DECISION_ANALYSIS.md §1.2.1）：
    - 使用 Gradio 5.45.0（独立 venv ``envs/voice-indextts2/``）
    - theme 在 ``Blocks`` 构造器中设置（Gradio 5.x 风格）
    - 与 VoxCPM 的 Gradio 6.0+ 通过 venv 隔离，无冲突
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import gradio as gr
import httpx

# v0.3.5+ 引擎子进程通过 SDK VoiceLibraryClient 访问平台音色库
# 详见 INDEXTTS2_SDK_VOICE_LIBRARY_ANALYSIS.md
# v0.4.4+：仅用于加载音色下拉（不暴露管理 Tab）
from aireliable_engine import VoiceLibraryClient

logger = logging.getLogger(__name__)


# ===========================================================================
# 常量
# ===========================================================================

#: IndexTTS2 支持的合成模式（v0.1 仅前 3 种）
_MODES = ["clone", "emo_clone", "emo_vector"]
_MODE_LABELS = {
    "clone": "克隆（仅音色参考音频）",
    "emo_clone": "情感克隆（音色 + 情感参考音频）",
    "emo_vector": "情感向量（8 维向量控制）",
}

#: 8 维情感标签（顺序与 IndexTTS2.infer() 一致，schemas.py EMOTION_LABELS）
_EMOTION_LABELS = [
    "开心 (happy)", "愤怒 (angry)", "悲伤 (sad)", "害怕 (afraid)",
    "厌恶 (disgusted)", "忧郁 (melanchoric)", "惊讶 (surprised)", "平静 (calm)",
]

#: 生成参数默认值（与 IndexTTS2 webui.py 预设一致）
_DEFAULT_TOP_P = 0.8
_DEFAULT_TOP_K = 30
_DEFAULT_TEMPERATURE = 0.8
_DEFAULT_REPETITION_PENALTY = 10.0
_DEFAULT_MAX_MEL_TOKENS = 1500
_DEFAULT_MAX_TEXT_TOKENS_PER_SEGMENT = 120
_DEFAULT_EMOTION_ALPHA = 1.0

#: HTTP 请求超时（秒）：合成耗时较长，给 300s
_HTTP_TIMEOUT = 300


# ===========================================================================
# API 调用辅助函数
# ===========================================================================

def _http_get(url: str, *, params: dict | None = None, timeout: float = 30.0) -> dict:
    """同步 GET 请求，返回 JSON。

    用于调 engine /capabilities、/status、gateway /v1/voices 等。
    """
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


def _fetch_voices(voice_client: VoiceLibraryClient) -> list[dict]:
    """拉取平台音色库列表（通过 SDK ``VoiceLibraryClient.list_voices_sync``）。

    v0.3.5+：从裸 httpx 改为 SDK 调用，与 PlatformTaskReporter 模式一致。
    SDK 内部已处理容错（gateway 不可达时返回空列表）+ DEBUG 日志（URL/响应码/耗时）。

    Args:
        voice_client: SDK 音色库客户端实例

    Returns:
        音色列表，每项含 id / name / sha256 / description / size / duration。
        失败时返回空列表（UI 显示友好提示）。
    """
    return voice_client.list_voices_sync()


def _fetch_voice_detail(voice_client: VoiceLibraryClient, voice_id: str) -> Optional[dict]:
    """查询单个音色详情（通过 SDK ``VoiceLibraryClient.get_voice_sync``）。

    用于合成时取音色文件本地路径（``path`` 字段）。

    Args:
        voice_client: SDK 音色库客户端实例
        voice_id: 音色 ID

    Returns:
        音色详情 dict，失败时返回 None
    """
    return voice_client.get_voice_sync(voice_id)


def _stream_infer(
    engine_url: str,
    payload: dict,
    progress_callback=None,
) -> tuple[Optional[str], Optional[str], list[str], Optional[str]]:
    """调 engine ``POST /infer``，读取 SSE 流，返回最终音频 URL / 本地路径。

    SSE 帧格式（PR/architecture.md §3.17A.2 + SDK sse.py）：
        - 普通进度帧: ``{"progress": 0.x, "stage": "xxx", "message": "...", ...}``
        - 完成帧（make_final_frame）: ``{"progress": 1.0, "stage": "completed", "result": {"audio_url": "/output/audio/xxx.wav", "audio_path": "D:\\...\\xxx.wav", ...}, "timing": {...}}``
        - 错误帧（make_error_frame）: ``{"progress": 0.0, "stage": "error", "error": {"type": "...", "message": "..."}}``

    关键修复（v0.4.4）：
        1. SDK 帧用 ``stage`` 字段标识类型，不是 ``type``（v0.4.4 修复）
        2. ``audio_url`` 嵌在 ``result`` 嵌套里（v0.4.4 修复）
        3. ``error.message`` 嵌在 ``error`` 嵌套里（v0.4.4 修复）
        4. 同时收集 ``audio_path``（绝对路径），供 Gradio Audio 控件本地播放

    Args:
        engine_url: 引擎根 URL（如 ``http://127.0.0.1:7005``）
        payload: InferRequest 请求体
        progress_callback: 可选回调 ``(message: str) -> None`` 用于实时显示进度

    Returns:
        (audio_url, audio_path, progress_messages, error_message)
        - audio_url: 成功时为 ``/output/audio/{task_id}.wav``（可拼 engine_url 下载，gateway 路径）
        - audio_path: 成功时为本地绝对路径（Gradio 播放用）
        - error_message: 失败时为错误描述，成功时为 None
    """
    audio_url: Optional[str] = None
    audio_path: Optional[str] = None
    messages: list[str] = []
    error_msg: Optional[str] = None

    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            with client.stream("POST", f"{engine_url}/infer", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    msg = data.get("message", "")
                    if msg:
                        messages.append(msg)
                        if progress_callback:
                            progress_callback(msg)

                    # v0.4.4 修复：SDK 用 stage 字段标识帧类型（make_final_frame 返回
                    # {"stage": "completed", ...}，make_error_frame 返回 {"stage": "error", ...}），
                    # 旧版 UI 错误用 data.get("type")，永远拿到 None，导致 UI 永远报"未返回音频 URL"。
                    frame_type = data.get("stage") or data.get("type")
                    if frame_type == "completed":
                        # v0.4.4 修复：audio_url 在 result 嵌套里
                        result = data.get("result", {})
                        if isinstance(result, dict):
                            audio_url = result.get("audio_url")
                            audio_path = result.get("audio_path")
                    elif frame_type == "error":
                        # v0.4.4 修复：error.message 在 error 嵌套里
                        err = data.get("error", {})
                        if isinstance(err, dict):
                            error_msg = err.get("message") or data.get("message", "未知错误")
                        else:
                            error_msg = msg or str(err) or "未知错误"
                        break
    except httpx.HTTPStatusError as e:
        error_msg = f"引擎返回错误 {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        error_msg = f"请求引擎失败: {e}"

    return audio_url, audio_path, messages, error_msg


def _resolve_audio_result(
    audio_url: Optional[str],
    audio_path: Optional[str],
    engine_url: str,
    success_label: str = "✅ 合成成功",
) -> tuple[Optional[str], str]:
    """从 ``_stream_infer`` 返回值中解析最终可播放路径（v0.4.4+）。

    优先 audio_path（Gradio 本地播放，绝对路径），降级 audio_url（拼 engine_url 完整 URL）。
    gateway 路径仍可下载（audio_url 返回给 gateway 任务上报）。

    Returns:
        (playable_path, status_message)
    """
    if audio_path and Path(audio_path).exists():
        return audio_path, f"{success_label}（本地播放）"
    if audio_url:
        full_url = audio_url if audio_url.startswith("http") else f"{engine_url}{audio_url}"
        return full_url, f"{success_label}（远程下载）"
    return None, "❌ 合成失败：未返回音频路径"


def _ensure_model_loaded(engine_url: str, variant: str) -> Optional[str]:
    """确保模型已加载，未加载则调 ``POST /load``。

    Returns:
        None 表示已加载或加载成功；非 None 字符串表示错误描述。
    """
    try:
        status = _http_get(f"{engine_url}/status", timeout=10.0)
        extra = status.get("extra", {})
        if extra.get("is_loaded") and extra.get("loaded_variant") == variant:
            return None
        # 需要加载
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            resp = client.post(
                f"{engine_url}/load",
                json={"variant": variant},
            )
            resp.raise_for_status()
        return None
    except Exception as e:
        return f"模型加载失败: {e}"


def _build_infer_payload(
    text: str,
    mode: str,
    variant: str,
    *,
    speaker_path: Optional[str] = None,
    emotion_path: Optional[str] = None,
    emotion_alpha: float = _DEFAULT_EMOTION_ALPHA,
    emotion_vector: Optional[list[float]] = None,
    emotion_text: Optional[str] = None,
    top_p: float = _DEFAULT_TOP_P,
    top_k: int = _DEFAULT_TOP_K,
    temperature: float = _DEFAULT_TEMPERATURE,
    repetition_penalty: float = _DEFAULT_REPETITION_PENALTY,
    max_mel_tokens: int = _DEFAULT_MAX_MEL_TOKENS,
    max_text_tokens_per_segment: int = _DEFAULT_MAX_TEXT_TOKENS_PER_SEGMENT,
) -> dict:
    """构造 InferRequest 请求体。

    详见 PR/architecture.md §3.17A.3 五种合成模式。
    """
    inputs: dict[str, Any] = {
        "text": text,
        "mode": mode,
        "top_p": top_p,
        "top_k": top_k,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "max_mel_tokens": max_mel_tokens,
        "max_text_tokens_per_segment": max_text_tokens_per_segment,
        "output_format": "wav",
    }
    if speaker_path:
        inputs["reference_audio_path"] = speaker_path
    if emotion_path:
        inputs["emotion_audio_path"] = emotion_path
        inputs["emotion_alpha"] = emotion_alpha
    if emotion_vector:
        inputs["emotion_vector"] = emotion_vector
    if emotion_text:
        inputs["emotion_text"] = emotion_text

    audios: list[dict] = []
    if speaker_path:
        audios.append({"path": speaker_path, "role": "speaker"})
    if emotion_path:
        audios.append({"path": emotion_path, "role": "emotion"})

    return {
        "variant": variant,
        "inputs": inputs,
        "audios": audios,
    }


# ===========================================================================
# Tab：情感控制（v0.4.4+ 单 Tab 化设计：覆盖原 基础 TTS + 克隆 + 情感控制）
# ===========================================================================

def _build_tab_emotion(engine_url: str, voice_client: VoiceLibraryClient) -> tuple:
    """Tab：情感控制（v0.4.4+ 单 Tab 化设计）。

    合并了原 v0.4.3 三个 Tab（基础 TTS / 克隆模式 / 情感控制），
    通过「合成模式」Radio 动态控制子区块显示/隐藏。

    v0.4.4+ 设计：
        - 顶部：平台音色库下拉 + 🔄 刷新按钮（自动填入 Audio 控件）
        - 音色参考：必填（从音色库选 或 直接上传）
        - 合成模式：Radio 4 选 1，根据选择动态展开对应参数区
            - clone       → 只需音色参考（基础克隆）
            - emo_clone   → 音色参考 + 情感参考音频 + α 权重
            - emo_vector  → 音色参考 + 8 维情感向量
            - emo_text    → 音色参考 + 情感描述文本（v0.2 实验性）
        - 高级参数：Accordion 折叠（top_p / top_k / temperature / ...）
        - 合成文本：Textbox
        - 生成 + 合成结果 Audio

    Returns:
        (refresh_fn, refresh_outputs)：供 create_ui 注册页面加载事件
    """
    gr.Markdown(
        "### 情感控制（单 Tab 化 v0.4.4+）\n"
        "IndexTTS2 零样本克隆 + 情感控制。\n\n"
        "**使用流程**：\n"
        "1. 选择合成模式（clone / emo_clone / emo_vector / emo_text）\n"
        "2. 上方选择平台音色（自动填入下方控件）或直接上传参考音频\n"
        "3. 根据模式补充对应参数（情感参考音频 / 8 维向量 / 文本描述）\n"
        "4. 输入合成文本 → 点击「🎵 生成语音」\n\n"
        "音色管理请到 **平台音色库管理页**（引擎 UI 不再重复暴露）。"
    )

    with gr.Row():
        voice_dropdown = gr.Dropdown(
            label="选择音色（平台音色库）",
            choices=[],
            value=None,
            interactive=True,
            info="选中后自动填入下方「音色参考音频」控件",
        )
        refresh_btn = gr.Button("🔄 刷新音色", variant="secondary", scale=0)

    with gr.Row():
        with gr.Column(scale=1):
            speaker_audio = gr.Audio(
                label="音色参考音频（必填）",
                type="filepath",
                format="wav",
            )
        with gr.Column(scale=1):
            mode_selector = gr.Radio(
                label="合成模式",
                choices=[(_MODE_LABELS[m], m) for m in _MODES],
                value="clone",
                info="clone=仅音色；emo_clone=音色+情感参考；emo_vector=8 维情感向量；emo_text=自然语言描述",
            )

    # ---- 模式专属子区块：默认只显示音色参考，Radio 切换时动态显示/隐藏 ----
    with gr.Group(visible=False) as emo_clone_group:
        gr.Markdown("**情感克隆模式（emo_clone）**：音色参考 + 情感参考音频 + 情感权重")
        with gr.Row():
            with gr.Column(scale=1):
                emotion_audio = gr.Audio(
                    label="情感参考音频（emo_clone 模式必填）",
                    type="filepath",
                    format="wav",
                )
            with gr.Column(scale=1):
                emotion_alpha = gr.Slider(
                    0.0, 1.0, value=_DEFAULT_EMOTION_ALPHA,
                    label="情感权重 (alpha)",
                    info="0.0=无情感，1.0=完全情感",
                )

    with gr.Group(visible=False) as emo_vector_group:
        gr.Markdown(
            "**情感向量模式（emo_vector）**：音色参考 + 8 维情感向量\n"
            "对应：开心/愤怒/悲伤/害怕/厌恶/忧郁/惊讶/平静"
        )
        emotion_sliders = []
        with gr.Row():
            for i in range(4):
                emotion_sliders.append(
                    gr.Slider(0.0, 1.0, value=0.0, label=_EMOTION_LABELS[i])
                )
        with gr.Row():
            for i in range(4, 8):
                emotion_sliders.append(
                    gr.Slider(0.0, 1.0, value=0.0, label=_EMOTION_LABELS[i])
                )

    with gr.Group(visible=False) as emo_text_group:
        gr.Markdown(
            "**情感文本模式（emo_text，v0.2 实验性）**：音色参考 + 情感描述文本\n"
            "⚠️ 依赖 Qwen3 微调模型，v0.1 可能未启用"
        )
        emotion_text = gr.Textbox(
            label="情感描述文本",
            placeholder="用自然语言描述情感（如：用愤怒的语气大声说）...",
            lines=2,
            interactive=True,
        )

    # ---- 高级参数（所有模式通用，Accordion 折叠）----
    with gr.Accordion("高级参数（可选）", open=False):
        with gr.Row():
            top_p = gr.Slider(0.0, 1.0, value=_DEFAULT_TOP_P, label="top_p")
            top_k = gr.Slider(0, 100, value=_DEFAULT_TOP_K, step=1, label="top_k")
            temperature = gr.Slider(0.0, 2.0, value=_DEFAULT_TEMPERATURE, label="temperature")
        with gr.Row():
            repetition_penalty = gr.Slider(0.0, 50.0, value=_DEFAULT_REPETITION_PENALTY, label="repetition_penalty")
            max_mel_tokens = gr.Slider(100, 3000, value=_DEFAULT_MAX_MEL_TOKENS, step=100, label="max_mel_tokens")
            max_text_tokens = gr.Slider(10, 300, value=_DEFAULT_MAX_TEXT_TOKENS_PER_SEGMENT, step=10, label="单段最大文本 token")

    text_input = gr.Textbox(
        label="合成文本",
        placeholder="请输入要合成的文本...",
        lines=4,
    )

    with gr.Row():
        generate_btn = gr.Button("🎵 生成语音", variant="primary")
        status_box = gr.Textbox(label="状态", interactive=False, scale=2)

    audio_output = gr.Audio(label="合成结果", type="filepath", format="wav")

    # 音色列表状态（存 name → 完整 card）
    voice_state = gr.State({})

    # ---- Radio change：动态切换子区块可见性 ----
    def _on_mode_change(mode):
        """v0.4.4+ 模式切换：根据 mode 决定 3 个 Group 的可见性。"""
        return {
            emo_clone_group: gr.update(visible=(mode == "emo_clone")),
            emo_vector_group: gr.update(visible=(mode == "emo_vector")),
            emo_text_group: gr.update(visible=(mode == "emo_text")),
        }

    mode_selector.change(
        _on_mode_change,
        inputs=[mode_selector],
        outputs=[emo_clone_group, emo_vector_group, emo_text_group],
    )

    def _refresh_voices():
        """v0.4.3+：mapping 存 name → 完整 card（含 path + audio_url）。"""
        voices = _fetch_voices(voice_client)
        if not voices:
            return gr.update(choices=[], value=None), {}, "⚠️ 音色库为空或 gateway 不可达"
        mapping = {v["name"]: v for v in voices if v.get("id") and v.get("name")}
        choices = list(mapping.keys())
        return (
            gr.update(choices=choices, value=choices[0] if choices else None),
            mapping,
            f"✅ 已加载 {len(voices)} 个音色",
        )

    refresh_btn.click(
        _refresh_voices,
        outputs=[voice_dropdown, voice_state, status_box],
    )

    # v0.4.3+ 修复：从 card dict 读本地路径
    def _select_voice(voice_name: str, mapping: dict):
        if not voice_name or voice_name not in mapping:
            return None, "⚠️ 请先选择音色"
        card = mapping[voice_name]
        local_path = card.get("path")
        if not local_path or not Path(local_path).exists():
            return None, "❌ 音色本地文件不可用（请到平台音色库管理页检查）"
        return local_path, f"✅ 已加载音色: {voice_name}"

    voice_dropdown.change(
        _select_voice,
        inputs=[voice_dropdown, voice_state],
        outputs=[speaker_audio, status_box],
    )

    def _generate(
        speaker_path, mode, emotion_path, alpha,
        e0, e1, e2, e3, e4, e5, e6, e7, emo_text_val, text,
        p, k, temp, rp, mmt, mtt,
    ):
        if not speaker_path:
            return None, "❌ 请先从音色库选择音色 或 直接上传参考音频"
        if not text.strip():
            return None, "❌ 请输入文本"

        # 模式校验
        if mode == "emo_clone" and not emotion_path:
            return None, "❌ emo_clone 模式需要情感参考音频"
        if mode == "emo_vector":
            vec = [e0, e1, e2, e3, e4, e5, e6, e7]
            if all(v == 0.0 for v in vec):
                return None, "❌ emo_vector 模式需要至少一个非零情感值"

        try:
            caps = _http_get(f"{engine_url}/capabilities", timeout=10.0)
            variants = caps.get("variants", [])
            if not variants:
                return None, "❌ 无可用模型变体，请先下载模型"
            variant = variants[0]["name"]
        except Exception as e:
            return None, f"❌ 获取能力声明失败: {e}"

        load_err = _ensure_model_loaded(engine_url, variant)
        if load_err:
            return None, f"❌ {load_err}"

        emotion_vector = [e0, e1, e2, e3, e4, e5, e6, e7] if mode == "emo_vector" else None
        payload = _build_infer_payload(
            text=text, mode=mode, variant=variant,
            speaker_path=speaker_path,
            emotion_path=emotion_path if mode == "emo_clone" else None,
            emotion_alpha=alpha,
            emotion_vector=emotion_vector,
            emotion_text=emo_text_val if mode == "emo_text" else None,
            top_p=p, top_k=k, temperature=temp,
            repetition_penalty=rp, max_mel_tokens=mmt,
            max_text_tokens_per_segment=mtt,
        )
        audio_url, audio_path, _, error = _stream_infer(engine_url, payload)
        if error:
            return None, f"❌ 合成失败: {error}"
        return _resolve_audio_result(audio_url, audio_path, engine_url, f"✅ {mode} 模式合成成功")

    generate_btn.click(
        _generate,
        inputs=[
            speaker_audio, mode_selector, emotion_audio, emotion_alpha,
            *emotion_sliders, emotion_text, text_input,
            top_p, top_k, temperature, repetition_penalty, max_mel_tokens, max_text_tokens,
        ],
        outputs=[audio_output, status_box],
    )

    # 返回 demo.load 需要的回调与输出，供 create_ui 注册页面加载事件
    return _refresh_voices, [voice_dropdown, voice_state, status_box]


# ===========================================================================
# create_ui 主函数（v0.4.4+ 单 Tab 化）
# ===========================================================================

def create_ui(
    engine_url: str,
    gateway_url: str,
    theme: Optional[Any] = None,
) -> gr.Blocks:
    """创建 IndexTTS2 Gradio Web UI（v0.4.4+ 单 Tab 化）。

    Args:
        engine_url: 引擎 API 根 URL（如 ``http://127.0.0.1:7005``），
                    UI 通过 httpx 调用 ``/capabilities`` / ``/load`` / ``/infer`` 等。
        gateway_url: gateway 根 URL（如 ``http://127.0.0.1:8000``），
                     v0.3.5+ 用于初始化 SDK ``VoiceLibraryClient``，
                     仅用于加载音色下拉（不暴露音色库 Tab）。
        theme: Gradio 主题（Gradio 5.45.0 在 Blocks 构造器中设置）。
               None 时使用默认 Soft 主题。

    Returns:
        gr.Blocks: 配置好的 Blocks 实例，由 server.py 的 ``mount_gradio_app`` 挂载。

    v0.4.4+ 单 Tab 化（用户决策）：
        - 合并 4 Tab → 1 Tab（情感控制）
        - 删除基础 TTS Tab（与 clone 模式重复）
        - 删除克隆模式 Tab（与 clone 模式重复）
        - 删除音色库 Tab（音色管理走平台音色库管理页）
        - 高级参数收纳到 Accordion
        - 通过 Radio 合成模式动态展开对应子区块（gr.Group visible）

    规范（PR/development-guide.md §28.20A.5）：
        - UI 通过 httpx 调自身 API（engine_url）
        - 音色库调用通过 SDK ``VoiceLibraryClient``（仅加载下拉用）
        - 页面加载时 ``demo.load()`` 自动刷新音色列表
    """
    if theme is None:
        theme = gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="slate",
            radius_size="lg",
        )

    # v0.3.5+ 初始化 SDK VoiceLibraryClient（仅用于加载音色下拉）
    voice_client = VoiceLibraryClient(gateway_url=gateway_url)
    logger.info(
        "VoiceLibraryClient 已初始化: gateway_url=%s",
        voice_client.gateway_url,
    )

    with gr.Blocks(
        title="IndexTTS2 TTS Engine",
        theme=theme,
        css="""
        .gradio-container { max-width: 1200px !important; }
        """,
    ) as demo:
        gr.Markdown(
            "# IndexTTS2 TTS Engine\n"
            "零样本克隆 + 情感控制（clone / emo_clone / emo_vector / emo_text）\n\n"
            f"- 引擎 API: `{engine_url}`\n"
            f"- 音色库 API: `{gateway_url}`（通过 SDK VoiceLibraryClient 调用音色下拉）\n"
            "- 音色管理请到 **平台音色库管理页**\n"
            "- 音色跨引擎复用（VoxCPM / IndexTTS2 共享平台音色库）"
        )

        # v0.4.4+ 单 Tab 化：直接调用 _build_tab_emotion，不再用 gr.Tabs 包裹
        emo_refresh_fn, emo_refresh_outputs = _build_tab_emotion(
            engine_url, voice_client,
        )

        # 页面加载时自动刷新音色下拉（参考 voxcpm/ui.py）
        demo.load(
            fn=emo_refresh_fn,
            outputs=emo_refresh_outputs,
        )

    return demo
