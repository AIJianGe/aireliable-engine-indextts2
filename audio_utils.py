"""IndexTTS2 音频工具（v0.4.3+ 引入）。

参考：engines/voxcpm_tts_c4a8f2e7/audio_utils.py
采样率：IndexTTS2 统一 24000Hz（VoxCPM 为 48000/16000）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_audio(
    samples: np.ndarray,
    sample_rate: int,
    output_path: Path,
    format: str = "wav",
) -> Path:
    """保存音频为文件。

    Args:
        samples: numpy array, shape (N,) 或 (N, C)
        sample_rate: 采样率（IndexTTS2=24000）
        output_path: 输出路径
        format: 格式（wav/flac，mp3 不支持时回退 wav）

    Returns:
        实际保存路径
    """
    import soundfile as sf

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 确保 float32
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32)

    # 归一化到 [-1, 1]
    max_val = float(np.max(np.abs(samples))) if samples.size > 0 else 0.0
    if max_val > 1.0:
        samples = samples / max_val

    # 保存（wav/flac 用 PCM_16）
    try:
        if format in ("wav", "flac"):
            sf.write(str(output_path), samples, sample_rate, subtype="PCM_16")
        else:
            # mp3 等不支持的格式回退 wav
            output_path = output_path.with_suffix(".wav")
            sf.write(str(output_path), samples, sample_rate, subtype="PCM_16")
    except Exception:
        # 兜底：直接写 wav
        output_path = output_path.with_suffix(".wav")
        sf.write(str(output_path), samples, sample_rate, subtype="PCM_16")

    return output_path


def get_audio_duration(samples: np.ndarray, sample_rate: int) -> float:
    """计算音频时长（秒）。

    Args:
        samples: numpy array, shape (N,) 或 (N, C)
        sample_rate: 采样率

    Returns:
        时长（秒）
    """
    if samples.ndim == 1:
        return len(samples) / sample_rate
    return samples.shape[-1] / sample_rate


def normalize_audio(samples: np.ndarray) -> np.ndarray:
    """归一化音频到 [-1, 1]。"""
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32)
    max_val = float(np.max(np.abs(samples))) if samples.size > 0 else 0.0
    if max_val > 0:
        samples = samples / max_val
    return samples
