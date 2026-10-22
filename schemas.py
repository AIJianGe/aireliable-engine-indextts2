"""IndexTTS2 引擎数据模型（v0.4.3+ 引入）。

定义 IndexTTS2 的输入/能力/状态数据模型。
参考：engines/voxcpm_tts_c4a8f2e7/schemas.py
规范：PR/architecture.md §3.17A / PR/development-guide.md §28.20A.2
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ===========================================================================
# 常量
# ===========================================================================

#: IndexTTS2 支持的合成模式
INDEXTTS2_MODES = ("clone", "emo_clone", "emo_vector", "emo_text")

#: 8 维情感向量标签（顺序与 IndexTTS2.infer() 一致）
EMOTION_LABELS = (
    "happy",        # 开心
    "angry",        # 愤怒
    "sad",          # 悲伤
    "afraid",       # 害怕
    "disgusted",    # 厌恶
    "melanchoric",  # 忧郁
    "surprised",    # 惊讶
    "calm",         # 平静
)

#: IndexTTS2 统一采样率
SAMPLE_RATE = 24000


# ===========================================================================
# 输入模型
# ===========================================================================

class IndexTTS2Inputs(BaseModel):
    """IndexTTS2 推理输入参数。

    字段映射（gateway → engine）：
        - extra.voice_id → audios[role=speaker].path → reference_audio_path
        - extra.audios[role=emotion] → emotion_audio_path
        - extra.emotion_alpha → emotion_alpha
        - extra.emotion_vector → emotion_vector
        - extra.emotion_text → emotion_text
        - extra.workflow → mode（gateway 同时写入两者）

    详见 PR/development-guide.md §28.25.2 字段映射对照表。
    """

    # 必填
    text: str = Field(..., description="合成文本")
    mode: str = Field("clone", description="合成模式")

    # clone 模式：参考音频路径（兜底，正常从 audios[role=speaker] 提取）
    reference_audio_path: Optional[str] = Field(
        None, description="音色参考音频路径（兜底，gateway 已解析为 audios[role=speaker]）"
    )

    # emo_clone 模式：情感参考音频 + 权重
    emotion_audio_path: Optional[str] = Field(
        None, description="情感参考音频路径（兜底，正常从 audios[role=emotion] 提取）"
    )
    emotion_alpha: float = Field(
        1.0, ge=0.0, le=1.0,
        description="情感权重（0.0=无情感，1.0=完全情感，默认 1.0）"
    )

    # emo_vector 模式：8 维情感向量
    emotion_vector: Optional[list[float]] = Field(
        None, min_length=8, max_length=8,
        description="8 维情感向量 [happy, angry, sad, afraid, disgusted, melanchoric, surprised, calm]"
    )

    # emo_text 模式：情感描述文本（v0.2 启用）
    emotion_text: Optional[str] = Field(
        None, description="情感描述文本（依赖 Qwen3 微调模型，v0.2 启用）"
    )

    # 生成参数（与 IndexTTS2 webui.py 预设一致）
    do_sample: bool = Field(True, description="是否采样")
    top_p: float = Field(0.8, ge=0.0, le=1.0, description="nucleus sampling")
    top_k: int = Field(30, ge=0, description="top-k sampling")
    temperature: float = Field(0.8, ge=0.0, description="采样温度")
    repetition_penalty: float = Field(10.0, ge=0.0, description="重复惩罚")
    max_mel_tokens: int = Field(1500, ge=1, description="最大 mel token 数")
    max_text_tokens_per_segment: int = Field(
        120, ge=1, description="单段最大文本 token"
    )

    # 兼容字段（gateway 同时写入 workflow 和 mode）
    workflow: Optional[str] = Field(None, description="合成模式（与 mode 同义，gateway 兼容字段）")

    # 输出
    output_format: str = Field("wav", description="输出音频格式")
    seed: int = Field(-1, ge=-1, description="随机种子（-1=随机）")


class IndexTTS2Capability(BaseModel):
    """IndexTTS2 引擎能力声明（variant 级别）。"""

    name: str = Field(..., description="variant 名称（author/model 格式）")
    alias: str = Field(..., description="variant 别名")
    precision: str = Field("fp16", description="精度")
    vram_required: int = Field(0, description="所需显存（MB）")
    sample_rate: int = Field(SAMPLE_RATE, description="采样率")
    languages: list[str] = Field(
        default_factory=lambda: ["zh", "en"],
        description="支持语言"
    )
    modes: list[str] = Field(
        default_factory=lambda: list(INDEXTTS2_MODES[:3]),
        description="支持模式（v0.1 仅 clone/emo_clone/emo_vector）"
    )
    emotion_labels: list[str] = Field(
        default_factory=lambda: list(EMOTION_LABELS),
        description="8 维情感标签"
    )
    loaded: bool = Field(False, description="是否已加载")


class IndexTTS2StatusExtra(BaseModel):
    """IndexTTS2 状态扩展字段。"""

    device: str = Field("cpu", description="设备")
    device_name: str = Field("", description="GPU 名称")
    vram_used: int = Field(0, description="已用显存（MB）")
    sample_rate: int = Field(SAMPLE_RATE, description="采样率")
    is_loaded: bool = Field(False, description="模型是否已加载")
    loaded_variant: Optional[str] = Field(None, description="已加载的 variant 名")
