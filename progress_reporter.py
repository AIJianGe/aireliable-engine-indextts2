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

v0.3.4+ 集成 SDK 任务上报：
    - emit_start → fire_report_running（gateway status=running）
    - emit (通用) → fire_report_progress（gateway 更新进度）
    - emit_completed → fire_report_succeeded（gateway status=succeeded）
    - emit_error → fire_report_failed（gateway status=failed）
    - 引擎 _infer_impl 入口调 report_created（gateway status=queued）
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from aireliable_engine import (
    SSEStream,
    make_frame,
    make_final_frame,
    make_error_frame,
    PlatformTaskReporter,
    get_platform_task_reporter,
)

logger = logging.getLogger(__name__)


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


class ProgressReporter:
    """IndexTTS2 进度上报器。

    封装 SSEStream，提供按阶段构造帧的便捷方法。
    v0.3.4+ 同时通过 SDK PlatformTaskReporter 上报任务状态到 gateway。
    """

    def __init__(self, stream: SSEStream, task_id: str = ""):
        """初始化进度上报器。

        Args:
            stream: SSE 流
            task_id: 任务 ID（v0.3.4+，用于 SDK 上报；为空则不上报）
        """
        self.stream = stream
        self.task_id = task_id
        self._start_time: float = time.time()
        self._task_reporter: Optional[PlatformTaskReporter] = None

    @property
    def elapsed(self) -> float:
        """已用时（秒）。"""
        return time.time() - self._start_time

    # ===== v0.3.4+ SDK 任务上报集成 =====

    async def init_task_reporter(self) -> None:
        """v0.3.4+ 异步初始化 SDK 任务上报器（捕获主 loop + 获取单例）。

        必须在 emit_* 方法之前调用（在 _infer_impl 入口处）。
        失败不阻断推理。
        """
        if self._task_reporter is None and self.task_id:
            try:
                self._task_reporter = await get_platform_task_reporter()
            except Exception as e:
                logger.warning("init_task_reporter 失败（不阻断推理）: %s", e)

    async def report_created(
        self,
        request_summary: dict,
        variant: str = "",
    ) -> None:
        """v0.3.4+ 上报任务创建到 gateway（status=queued）。"""
        if self._task_reporter and self.task_id:
            try:
                await self._task_reporter.report_created(
                    task_id=self.task_id,
                    request_summary=request_summary,
                    variant=variant,
                )
            except Exception as e:
                logger.warning("report_created 失败（不阻断推理）: %s", e)

    def _fire_running(self, stage: str) -> None:
        """v0.3.4+ fire-and-forget 上报任务进入 running 状态。"""
        if self._task_reporter and self.task_id:
            self._task_reporter.fire_report_running(self.task_id, stage=stage)

    def _fire_progress(self, progress: float, stage: str) -> None:
        """v0.3.4+ fire-and-forget 上报任务进度。"""
        if self._task_reporter and self.task_id:
            self._task_reporter.fire_report_progress(
                self.task_id, progress=progress, stage=stage,
            )

    def _fire_succeeded(self, result: dict) -> None:
        """v0.3.4+ fire-and-forget 上报任务成功。"""
        if self._task_reporter and self.task_id:
            self._task_reporter.fire_report_succeeded(self.task_id, result=result)

    def _fire_failed(self, error: dict) -> None:
        """v0.3.4+ fire-and-forget 上报任务失败。"""
        if self._task_reporter and self.task_id:
            self._task_reporter.fire_report_failed(self.task_id, error=error)

    # ===== 通用帧 =====

    def emit(self, progress: float, stage: str, **extra: Any) -> None:
        """发送一帧进度（线程安全）。"""
        frame = make_frame(progress, stage, **extra)
        self.stream.put_threadsafe(frame)
        self._fire_progress(progress, stage)

    async def emit_async(self, progress: float, stage: str, **extra: Any) -> None:
        """发送一帧进度（协程内调用）。"""
        frame = make_frame(progress, stage, **extra)
        await self.stream.put(frame)
        self._fire_progress(progress, stage)

    def emit_error(self, message: str, error_type: str = "indextts2_error") -> None:
        """发送错误帧。"""
        frame = make_error_frame(message, error_type)
        self.stream.put_threadsafe(frame)
        self._fire_failed({"type": error_type, "message": message})

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
        timing = {
            "total_seconds": self.elapsed,
            "audio_duration": duration,
            "sample_rate": sample_rate,
        }
        result = {
            "audio_url": audio_url,
            "audio_path": audio_path,
            "duration": duration,
            "sample_rate": sample_rate,
            "mode": mode,
        }
        frame = make_final_frame(result, timing)
        await self.stream.put(frame)
        self._fire_succeeded(result)

    def emit_completed_threadsafe(
        self,
        audio_url: str,
        duration: float,
        sample_rate: int,
        mode: str,
        audio_path: str = "",
    ) -> None:
        """完成（线程安全版本）。"""
        timing = {
            "total_seconds": self.elapsed,
            "audio_duration": duration,
            "sample_rate": sample_rate,
        }
        result = {
            "audio_url": audio_url,
            "audio_path": audio_path,
            "duration": duration,
            "sample_rate": sample_rate,
            "mode": mode,
        }
        frame = make_final_frame(result, timing)
        self.stream.put_threadsafe(frame)
        self._fire_succeeded(result)
