"""IndexTTS2 引擎配置（v0.4.3+ 引入）。

参考：engines/voxcpm_tts_c4a8f2e7/config.py
规范：PR/development-guide.md §28.20A.9 / project-rules.md §第三部分
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

# ===========================================================================
# 常量
# ===========================================================================

ENGINE_ID = "indextts2_b7c3d9e1"
ENGINE_NAME = "IndexTTS2-TTS"
CAPABILITY = "text-to-speech"
ENGINE_TYPE = "voice"
DEFAULT_PORT = 7005

#: 模型子目录（L1 层 voice/tts/indextts2/）
_DEFAULT_MODEL_SUBDIR = "voice/tts/indextts2/"


def _resolve_model_subdir() -> str:
    """v0.2.1 双源真相消除：优先读 AIRELIABLE_MODEL_LAYOUT_BASE 环境变量。

    平台通过 engine_manager._build_subprocess_env 注入此变量。
    详见 PR/architecture.md §3.6 L0 资源层 / project-rules.md §第三部分 §1。
    """
    env_val = os.environ.get("AIRELIABLE_MODEL_LAYOUT_BASE")
    if env_val:
        return env_val.rstrip("/") + "/"
    return _DEFAULT_MODEL_SUBDIR


MODEL_SUBDIR = _resolve_model_subdir()


# ===========================================================================
# EngineConfig
# ===========================================================================

@dataclass
class EngineConfig:
    """IndexTTS2 引擎配置。

    字段对齐 VoxCPM EngineConfig，适配 IndexTTS2 参数：
        - port: 7005（VoxCPM=7003，FoleyCrafter=7004，递增）
        - model_dir: models/voice/tts/indextts2/
        - log_dir: logs/engines/（v0.2.15+ 统一，依赖 AIRELIABLE_LOG_DIR）
    """

    # 基础
    engine_id: str = ENGINE_ID
    engine_name: str = ENGINE_NAME
    capability: str = CAPABILITY
    engine_type: str = ENGINE_TYPE
    port: int = DEFAULT_PORT
    host: str = "127.0.0.1"

    # 路径
    models_root: Path = Path("models")
    data_root: Path = Path("data")
    project_root: Path = Path(".")
    output_root: Path = Path("output")

    # 推导路径（init=False）
    model_dir: Path = field(init=False)
    log_dir: Path = field(init=False)

    # 设备
    device: str = "auto"

    # 日志
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.model_dir = self.models_root / MODEL_SUBDIR
        # v0.2.15+ 依赖 AIRELIABLE_LOG_DIR 环境变量推导 log_dir
        env_log_dir = os.environ.get("AIRELIABLE_LOG_DIR")
        if env_log_dir:
            self.log_dir = Path(env_log_dir)
        else:
            self.log_dir = self.project_root / "logs" / "engines"
        # 确保目录存在
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# 命令行解析
# ===========================================================================

def parse_args() -> EngineConfig:
    """解析命令行参数 + 环境变量。

    优先级：命令行 > 环境变量 > 默认值

    环境变量（平台注入，详见 project-rules.md §第三部分 §1）：
        - AIRELIABLE_MODELS_ROOT: models 根目录
        - AIRELIABLE_MODEL_LAYOUT_BASE: 当前引擎的 model_layout.base
        - AIRELIABLE_LOG_DIR: 引擎日志目录
        - AIRELIABLE_CORE_URL: 平台 API 地址
    """
    parser = argparse.ArgumentParser(description="IndexTTS2 TTS 引擎")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"端口号（默认 {DEFAULT_PORT}）")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--device", type=str, default="auto",
                        help="设备（auto/cpu/cuda，默认 auto）")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="日志级别（默认 INFO）")
    parser.add_argument("--models-root", type=str, default=None,
                        help="models 根目录（默认读 AIRELIABLE_MODELS_ROOT）")
    args = parser.parse_args()

    models_root = Path(
        args.models_root
        or os.environ.get("AIRELIABLE_MODELS_ROOT", "models")
    )
    return EngineConfig(
        port=args.port,
        host=args.host,
        device=args.device,
        log_level=args.log_level,
        models_root=models_root,
    )
