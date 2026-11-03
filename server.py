"""IndexTTS2 TTS 引擎 HTTP 服务（FastAPI + 原生 WebUI 单端口）。

8 个标准端点与 CORS / 日志级别 / 根路径重定向 / 静态挂载由 SDK
``build_engine_app`` 提供，本文件只保留引擎特有部分：

    GET  /output/*   静态音频文件（推理产物）
    GET  /voices     平台音色库代理（转发 gateway GET /v1/voices，WebUI 同源调用）
    POST /upload     参考音频上传（WebUI 自定义音色/情感参考，返回引擎侧路径）
    GET  /ui/*       原生 WebUI 静态文件（webui/ 三件套，FLUX 风格）

启动（参考 PR/development-guide.md §28.20A.8）::

    uv run --no-sync --project envs/voice-indextts2 \\
        python -m engines.indextts2_b7c3d9e1.server --port 7005

WebUI 说明（v0.5.0+ 引擎 WebUI 风格统一）：
    - Gradio UI（ui.py）已废弃，改为原生 HTML/CSS/JS 三件套（webui/）
    - 音色库数据源：WebUI 同源调 ``GET /voices``（本文件代理转发 gateway），
      不再依赖 SDK VoiceLibraryClient（那是 Gradio 服务端渲染时代的方案）
    - 参考音频经 ``POST /upload`` 落盘后以 ``audios[role=speaker|emotion].path``
      传入 ``/infer``（与 gateway 解析 voice_id 后的请求结构一致）

规范：project-rules.md §第二部分 §6（日志规范）/ §第三部分 §1（环境变量）
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, FastAPI

from aireliable_engine import build_engine_app, mount_upload, run_engine_server

from .config import EngineConfig, parse_args
from .engine import IndexTTS2Engine

logger = logging.getLogger(__name__)

WEBUI_DIR = Path(__file__).parent / "webui"

# 由 create_app 写入，供 /upload 在请求时解析落盘目录（import 时配置尚未解析）
_config: Optional[EngineConfig] = None

# ===== 引擎特有路由（静态挂载之后注册）=====
router = APIRouter()


def _upload_dir() -> Path:
    """上传目录（每次请求时按当前配置解析）。"""
    if _config is None:
        raise RuntimeError("配置未初始化")
    return _config.output_root / "uploads"


mount_upload(router, _upload_dir, log_label="参考音频")


@router.get("/voices")
async def voices() -> list:
    """转发 gateway ``GET /v1/voices``，返回平台音色列表。

    gateway 原始响应为 ``{"data": [...], "total": N}`` 包装结构，
    此处解包为纯数组，并注入 ``audio_url``（gateway 原始 card 不含此字段，
    WebUI 预览播放用）。gateway 不可达时返回空列表。
    """
    gateway_url = _resolve_gateway_url()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{gateway_url}/v1/voices")
            resp.raise_for_status()
            data = resp.json()
        voices_list = data if isinstance(data, list) else data.get("data", [])
        # 富化 audio_url（浏览器预览播放；<audio> 标签跨源加载不受 CORS 限制）
        for v in voices_list:
            if isinstance(v, dict) and v.get("id") and not v.get("audio_url"):
                v["audio_url"] = f"{gateway_url}/v1/voices/{v['id']}/audio"
        logger.info("音色库代理: %s 返回 %d 个音色", gateway_url, len(voices_list))
        return voices_list
    except Exception as e:
        # 音色库不可用不阻断引擎，WebUI 端已有空列表容错
        logger.warning("音色库代理失败（gateway=%s）: %s", gateway_url, e)
        return []


def _resolve_gateway_url() -> str:
    """解析 gateway URL（用于 WebUI 调用平台音色库 API）。

    优先级：
        1. ``AIRELIABLE_GATEWAY_URL``（平台注入，见 ``engine_manager._build_subprocess_env``）
        2. ``AIRELIABLE_CORE_URL``（兼容旧配置的降级兜底）
        3. 默认 ``http://127.0.0.1:8000``（gateway 默认端口）

    注意：音色库 API 仅存在于 gateway（8000），不是 core（8001）。
    """
    import os

    gw_url = os.environ.get("AIRELIABLE_GATEWAY_URL", "").strip()
    if gw_url:
        return gw_url.rstrip("/")
    core_url = os.environ.get("AIRELIABLE_CORE_URL", "").strip()
    if core_url:
        return core_url.rstrip("/")
    return "http://127.0.0.1:8000"


def create_app(config: EngineConfig | None = None) -> FastAPI:
    """创建 FastAPI 应用（配置缺省时从命令行解析）。"""
    global _config

    if config is None:
        config = parse_args()
    _config = config

    return build_engine_app(
        IndexTTS2Engine,
        config=config,
        title="IndexTTS2 TTS Engine",
        description=(
            "AIReliable 平台的 IndexTTS2 语音合成引擎\n"
            "（零样本克隆 / 情感控制：clone / emo_clone / emo_vector）\n\n"
            "音色库集成：通过 ``extra.voice_id`` 引用平台音色库，跨引擎复用。"
        ),
        version="0.1.0",
        # v0.2.11+ 方案 B：lifespan startup 阶段异步刷新扫描（平台 API 已就绪）
        startup_hook=lambda engine: engine._refresh_scan_async(),
        static_dirs={"/ui": WEBUI_DIR, "/output": config.output_root},
        extra_routers=[router],
    )


def main() -> None:
    """命令行启动入口。"""
    config = parse_args()
    app = create_app(config)
    run_engine_server(app, config=config)


# 模块级 app 实例（uvicorn 直接引用 ``engines.indextts2_b7c3d9e1.server:app`` 时需手动赋值）
app: FastAPI | None = None


if __name__ == "__main__":
    main()
