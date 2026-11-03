"""SSE 进度上报工具（IndexTTS2 TTS 专用，v0.4.3+ 引入）。

把 IndexTTS2 的推理过程转换为 AIReliable 进度帧。
参考：engines/voxcpm_tts_c4a8f2e7/progress_reporter.py

进度阶段：
    start             0.00          开始 TTS 推理
    model_ready       0.05          模型已加载确认
    text_processing   0.10          文本前端处理
    generating        0.15 - 0.90   生成音频中
    saving            0.90          保存音频文件
    completed         1.00          完成

任务 71 P0-B：通用帧构造 / 任务上报 / 心跳 / 完成帧全部由 SDK
``BaseProgressReporter`` 提供，本文件只保留 IndexTTS2 的阶段常量与语义化方法。
"""

from __future__ import annotations

from typing import Any

from aireliable_engine.progress import BaseProgressReporter


# ===== 进度阶段定义 =====

class ProgressStage:
    """IndexTTS2 推理进度阶段。"""

    START = "indextts2.start"                      # 0.00
    MODEL_READY = "indextts2.model_ready"           # 0.05
    TEXT_PROCESSING = "indextts2.text_processing"   # 0.10
    GENERATING = "indextts2.generating"             # 0.15 - 0.90
    SAVING = "indextts2.saving"                     # 0.90
    COMPLETED = "indextts2.completed"               # 1.00
    ERROR = "indextts2.error"


# 进度区间
STAGE_RANGES: dict[str, tuple[float, float]] = {
    ProgressStage.START: (0.00, 0.05),
    ProgressStage.MODEL_READY: (0.05, 0.10),
    ProgressStage.TEXT_PROCESSING: (0.10, 0.15),
    ProgressStage.GENERATING: (0.15, 0.90),
    ProgressStage.SAVING: (0.90, 1.00),
}


class ProgressReporter(BaseProgressReporter):
    """IndexTTS2 进度上报器（阶段语义 + SDK 通用帧）。"""

    error_type = "indextts2_error"

    # ===== 阶段便捷方法 =====

    def emit_start(self, text: str, mode: str) -> None:
        """开始推理。"""
        self.emit(
            0.0,
            ProgressStage.START,
            message="开始 IndexTTS2 推理",
            text=text[:100],
            mode=mode,
        )
        self._fire_running(ProgressStage.START)

    def emit_model_ready(self, variant: str, device: str) -> None:
        """模型已就绪。"""
        self.emit(
            0.05,
            ProgressStage.MODEL_READY,
            message="模型已加载",
            variant=variant,
            device=device,
        )

    def emit_text_processing(self, text: str) -> None:
        """文本处理中。"""
        self.emit(
            0.10,
            ProgressStage.TEXT_PROCESSING,
            message="文本前端处理",
            text_length=len(text),
        )

    def emit_generating(self, progress: float = 0.15, **extra: Any) -> None:
        """生成中。"""
        self.emit(progress, ProgressStage.GENERATING, message="生成音频中...", **extra)

    def emit_saving(self) -> None:
        """保存中。"""
        self.emit(0.90, ProgressStage.SAVING, message="保存音频文件")

    async def emit_completed(
        self,
        audio_url: str,
        duration: float,
        sample_rate: int,
        mode: str,
        audio_path: str = "",
    ) -> None:
        """完成。

        Args:
            audio_url: 音频 URL 路径（如 /output/audio/xxx.wav），用于下载链接
            audio_path: 音频本地绝对路径，用于 Gradio UI 播放（绕过 SSRF 防护）
        """
        await super().emit_completed(
            {
                "audio_url": audio_url,
                "audio_path": audio_path,
                "duration": duration,
                "sample_rate": sample_rate,
                "mode": mode,
            },
            {
                "total_seconds": self.elapsed,
                "audio_duration": duration,
                "sample_rate": sample_rate,
            },
        )

    def emit_completed_threadsafe(
        self,
        audio_url: str,
        duration: float,
        sample_rate: int,
        mode: str,
        audio_path: str = "",
    ) -> None:
        """完成（线程安全版本）。"""
        super().emit_completed_threadsafe(
            {
                "audio_url": audio_url,
                "audio_path": audio_path,
                "duration": duration,
                "sample_rate": sample_rate,
                "mode": mode,
            },
            {
                "total_seconds": self.elapsed,
                "audio_duration": duration,
                "sample_rate": sample_rate,
            },
        )
