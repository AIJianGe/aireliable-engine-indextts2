"""IndexTTS2 TTS 引擎 HTTP 服务（FastAPI + Gradio 5.45.0 单端口）。

实现 8 个标准 HTTP 端点 + 挂载 Gradio UI + 挂载输出目录。

启动（参考 PR/development-guide.md §28.20A.8）::

    uv run --no-sync --project envs/voice-indextts2 \\
        python -m engines.indextts2_b7c3d9e1.server --port 7005

端点：
    GET  /health         健康检查
    GET  /capabilities   能力声明（可用模型变体 / 支持模式 / 8 维情感标签）
    POST /load           加载 IndexTTS2 模型权重
    POST /unload         卸载模型（释放显存）
    POST /infer          TTS 推理（SSE 流式进度）
    GET  /status         状态查询
    POST /lora/load      v0.1 返回 501
    POST /lora/unload    v0.1 返回 501
    GET  /output/*       静态音频文件（推理产物）
    GET  /voices         平台音色库代理（转发 gateway GET /v1/voices，WebUI 同源调用）
    POST /upload         参考音频上传（WebUI 自定义音色/情感参考，返回引擎侧路径）
    GET  /               307 重定向到 /ui/
    GET  /ui/*           原生 WebUI 静态文件（webui/ 三件套，FLUX 风格）

WebUI 说明（v0.5.0+ 引擎 WebUI 风格统一）：
    - Gradio UI（ui.py）已废弃，改为原生 HTML/CSS/JS 三件套（webui/）
    - 风格基底与 FLUX 引擎一致（ENGINE_WEBUI_FLUX_STYLE_UNIFY 分析）
    - 音色库数据源：WebUI 同源调 ``GET /voices``（本文件代理转发 gateway），
      不再依赖 SDK VoiceLibraryClient（那是 Gradio 服务端渲染时代的方案）
    - 参考音频经 ``POST /upload`` 落盘后以 ``audios[role=speaker|emotion].path``
      传入 ``/infer``（与 gateway 解析 voice_id 后的请求结构一致）

规范：project-rules.md §第二部分 §6（日志规范）/ §第三部分 §1（环境变量）
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from aireliable_engine import (
    ErrorResponse,
    HealthResponse,
    InferRequest,
    LoadRequest,
    LoraLoadRequest,
    LoraUnloadRequest,
    register_log_level_endpoint,
    setup_engine_logging,
)

from .config import EngineConfig, parse_args
from .engine import IndexTTS2Engine

logger = logging.getLogger(__name__)


# ===== 全局引擎实例 =====
# FastAPI 启动时初始化，所有端点共享
_engine: Optional[IndexTTS2Engine] = None
_config: Optional[EngineConfig] = None


def get_engine() -> IndexTTS2Engine:
    """获取引擎实例。"""
    if _engine is None:
        raise RuntimeError("引擎未初始化（FastAPI lifespan 未执行？）")
    return _engine


def get_config() -> EngineConfig:
    """获取配置。"""
    if _config is None:
        raise RuntimeError("配置未初始化")
    return _config


# ===== Lifespan 生命周期管理 =====

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期管理。

    启动时初始化引擎，关闭时清理资源（卸载模型）。

    关键事件记录（project-rules.md §第二部分 §6.5）：
        - 引擎实例化 + variant 扫描路径 + 结果数量
        - 关闭时卸载模型结果
    """
    global _engine, _config

    # 解析配置（如果尚未从命令行注入）
    if _config is None:
        _config = parse_args()

    # 配置日志（v0.2.15+ 依赖 AIRELIABLE_LOG_DIR，由 config.py 推导 log_dir）
    setup_engine_logging(
        engine_id=_config.engine_id,
        log_level=_config.log_level,
        log_dir=_config.log_dir,
    )

    logger.info("===== IndexTTS2 TTS 引擎启动 =====")
    logger.info("配置: %s", _config.to_dict() if hasattr(_config, "to_dict") else vars(_config))

    # 初始化引擎
    _engine = IndexTTS2Engine(_config)
    # v0.2.11+ 方案 B：lifespan startup 阶段异步刷新扫描（平台 API 已就绪）
    # __init__ 阶段的同步 _refresh_scan 只是 fallback，此处用平台 API 重新检测
    await _engine._refresh_scan_async()
    logger.info(
        "引擎实例化完成: state=%s, variants=%d, default_variant=%s",
        _engine.state_machine.state.value,
        len(_engine._available_variants),
        _engine.default_variant,
    )

    yield

    # 关闭清理：卸载模型
    logger.info("===== IndexTTS2 TTS 引擎关闭 =====")
    if _engine is not None and _engine.wrapper.is_loaded:
        try:
            await _engine._unload_impl()
            logger.info("关闭时卸载模型成功")
        except Exception as e:
            logger.warning("关闭时卸载模型失败: %s", e)
    _engine = None


# ===== 创建 FastAPI 应用 =====

def create_app(config: EngineConfig | None = None) -> FastAPI:
    """创建 FastAPI 应用。

    Args:
        config: 引擎配置（None 则从命令行解析）

    Returns:
        FastAPI: 配置好的应用实例
    """
    global _config
    if config is not None:
        _config = config

    app = FastAPI(
        title="IndexTTS2 TTS Engine",
        description=(
            "AIReliable 平台的 IndexTTS2 语音合成引擎\n"
            "（零样本克隆 / 情感控制：clone / emo_clone / emo_vector）\n\n"
            "音色库集成：通过 ``extra.voice_id`` 引用平台音色库，跨引擎复用。"
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS（允许主控跨域调用，iframe 嵌入需要）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],                  # v0.1 简化，生产环境应限制
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ===== 注册 8 个标准端点 =====
    _register_endpoints(app)

    # ===== 挂载输出目录（/output/* 提供音频文件下载）=====
    if _config is not None:
        _config.output_root.mkdir(parents=True, exist_ok=True)
        app.mount(
            "/output",
            StaticFiles(directory=str(_config.output_root)),
            name="output",
        )
        logger.info("输出目录挂载: /output -> %s", _config.output_root)

        # ===== 挂载原生 WebUI（webui/ 三件套，FLUX 风格）=====
        _mount_webui(app)

    return app


def _register_endpoints(app: FastAPI) -> None:
    """注册 8 个标准 HTTP 端点 + 日志级别端点。

    详见 PR/architecture.md §3.17A.2 端点与请求格式。
    """

    # 1. /health
    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """健康检查。"""
        return await get_engine().health()

    # 2. /capabilities
    @app.get("/capabilities")
    async def capabilities() -> dict:
        """能力声明（可用模型变体 / 支持模式 / 8 维情感标签）。

        详见 PR/architecture.md §3.17A.1 引擎身份。
        """
        return (await get_engine().capabilities()).model_dump()

    # 3. /load
    @app.post("/load")
    async def load(request: LoadRequest) -> dict:
        """加载 IndexTTS2 模型权重。

        加载流程（project-rules.md §第二部分 §2.2 正确架构）：
            1. 文件完整性校验（不下载，只校验）
            2. 显存预检（FP16 需 ~6GB）
            3. torch.load monkey-patch（try/finally 包裹整个加载流程）
            4. 构造 aux_paths（绕过 ensure_models_available 自动下载）
            5. 调用 IndexTTS2.__init__() 加载模型
        """
        try:
            response = await get_engine().load(request)
            return response.model_dump()
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            # 文件缺失 / 显存不足 / 加载失败 → 500 + 详细信息
            logger.exception("加载失败")
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.exception("加载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 4. /unload
    @app.post("/unload")
    async def unload() -> dict:
        """卸载模型（释放显存）。"""
        try:
            response = await get_engine().unload()
            return response.model_dump()
        except Exception as e:
            logger.exception("卸载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 5. /infer（SSE 流式）
    @app.post("/infer")
    async def infer(request: InferRequest):
        """TTS 推理（SSE 流式返回进度 + 最终音频 URL）。

        请求体参考 InferRequest schema + PR/architecture.md §3.17A.3 五种合成模式：
            - variant: 模型变体名（如 ``IndexTeam/IndexTTS-2``）
            - inputs: IndexTTS2Inputs 字段（text / mode / emotion_vector / ...）
            - audios: 参考音频列表
                - ``audios[role=speaker]`` → spk_audio_prompt（音色参考，必填）
                - ``audios[role=emotion]`` → emo_audio_prompt（情感参考，可选）
            - task_id: 可选任务 ID

        模式决策（PR/development-guide.md §28.25.3）：
            - 有 emotion 音频 → 自动 emo_clone
            - 否则按 inputs.mode（clone / emo_vector / emo_text）
        """
        engine = get_engine()

        # BaseEngine.infer 返回异步迭代器（SSE 帧）
        sse_stream = engine.infer(request)

        return StreamingResponse(
            sse_stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",          # Nginx 不缓冲
            },
        )

    # 6. /status
    @app.get("/status")
    async def status() -> dict:
        """引擎状态查询。"""
        return (await get_engine().status()).model_dump()

    # 7. /lora/load（v0.1 返回 501）
    @app.post("/lora/load")
    async def lora_load(request: LoraLoadRequest) -> dict:
        """加载 LoRA（v0.1 不支持）。"""
        try:
            response = await get_engine().load_lora(request)
            return response.model_dump()
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("LoRA 加载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 8. /lora/unload（v0.1 返回 501）
    @app.post("/lora/unload")
    async def lora_unload(request: LoraUnloadRequest) -> dict:
        """卸载 LoRA（v0.1 不支持）。"""
        try:
            response = await get_engine().unload_lora(request)
            return response.model_dump()
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("LoRA 卸载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 错误处理
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """HTTP 异常统一格式。"""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error={
                "type": exc.__class__.__name__,
                "message": str(exc.detail),
                "code": exc.status_code,
            }).model_dump(),
        )

    # v0.2.5+ T6.2：日志级别动态调整端点（GET /log_level + POST /log_level）
    register_log_level_endpoint(app)

    # ===== v0.5.0+ WebUI 专属端点 =====

    # 根路径 → /ui/（307 重定向，浏览器打开引擎根地址直接进 WebUI）
    @app.get("/", include_in_schema=False)
    async def _root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/ui/", status_code=307)

    # 参考音频上传（WebUI 自定义音色/情感参考；分块写盘防大文件占内存）
    @app.post("/upload")
    async def upload(file: UploadFile) -> dict:
        """上传参考音频，落盘到 ``output/uploads/``，返回引擎侧绝对路径。

        WebUI 上传后以 ``audios[role=speaker|emotion].path`` 引用返回的
        ``path``（与 gateway 解析平台音色 voice_id 后的请求结构一致，
        引擎 ``_infer_impl`` 无需区分两种来源）。
        """
        config = get_config()
        upload_dir = config.output_root / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        # uuid 前缀防重名覆盖；Path().name 防路径穿越
        safe_name = Path(file.filename or "upload.bin").name
        dest = upload_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"
        size = 0
        try:
            with open(dest, "wb") as f:
                while True:
                    chunk = await file.read(1024 * 1024)  # 1MB 分块
                    if not chunk:
                        break
                    f.write(chunk)
                    size += len(chunk)
        finally:
            await file.close()
        logger.info("参考音频上传: %s (%d bytes) -> %s", safe_name, size, dest)
        return {"path": str(dest), "filename": safe_name, "size": size}

    # 平台音色库代理（WebUI 同源调用，避免浏览器跨域/端口发现）
    @app.get("/voices")
    async def voices() -> list:
        """转发 gateway ``GET /v1/voices``，返回平台音色列表。

        gateway 原始响应为 ``{"data": [...], "total": N}`` 包装结构，
        此处解包为纯数组，并按 SDK ``VoiceLibraryClient`` v0.4.3+ 富化约定
        注入 ``audio_url``（gateway 原始 card 不含此字段，WebUI 预览播放用）。
        gateway 不可达时返回空列表（WebUI 显示空下拉并提示）。
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


def _mount_webui(app: FastAPI) -> None:
    """挂载原生 WebUI 到 /ui（StaticFiles，FLUX 风格三件套）。

    v0.5.0+ 引擎 WebUI 风格统一（ENGINE_WEBUI_FLUX_STYLE_UNIFY）：
        - ``webui/`` 目录含 index.html / style.css / app.js
        - 根路径 ``/`` 由 ``_register_endpoints`` 中的 307 重定向跳到 ``/ui/``
        - ui.py（Gradio）已废弃，不再挂载
    """
    webui_dir = Path(__file__).parent / "webui"
    if webui_dir.is_dir() and (webui_dir / "index.html").exists():
        app.mount(
            "/ui",
            StaticFiles(directory=str(webui_dir), html=True),
            name="webui",
        )
        logger.info("原生 WebUI 挂载: /ui -> %s", webui_dir)
    else:
        logger.warning("WebUI 未挂载（webui/index.html 不存在）: %s", webui_dir)


def _resolve_gateway_url() -> str:
    """解析 gateway URL（用于 UI 调用平台音色库 API）。

    v0.2.16+ 修订：函数名是 ``_resolve_gateway_url``，但之前错读 ``AIRELIABLE_CORE_URL``
    （指向 core 8001 端口），而音色库 API 仅存在于 gateway（端口 8000）。
    结果 ``GET http://127.0.0.1:8001/v1/voices`` 返回 404，UI 下拉空。

    修复后优先级：
        1. ``AIRELIABLE_GATEWAY_URL`` 环境变量（平台注入，详见 ``engine_manager._build_subprocess_env``
           第 2065 行，指向 gateway 默认端口 8000）
        2. ``AIRELIABLE_CORE_URL`` 环境变量（兼容旧配置，理论上 core 应反代 /v1/voices
           到 gateway，但当前 core 无此端点，此分支仅作降级兜底）
        3. 默认 ``http://127.0.0.1:8000``（gateway 默认端口）
    """
    import os
    # 优先 gateway（音色库 API 仅在 gateway 暴露）
    gw_url = os.environ.get("AIRELIABLE_GATEWAY_URL", "").strip()
    if gw_url:
        return gw_url.rstrip("/")
    # 兜底：若仅有 core URL（core 反代场景，当前未实现）
    core_url = os.environ.get("AIRELIABLE_CORE_URL", "").strip()
    if core_url:
        return core_url.rstrip("/")
    # 最终默认：gateway 默认端口
    return "http://127.0.0.1:8000"


# ===== 入口 =====

# 模块级 app 实例（uvicorn 直接引用）
app: Optional[FastAPI] = None


def main() -> None:
    """命令行启动入口。"""
    global _config, app

    # 解析命令行参数
    _config = parse_args()

    # 配置日志（lifespan 也会配置，这里先配一遍让启动日志可见）
    setup_engine_logging(
        engine_id=_config.engine_id,
        log_level=_config.log_level,
        log_dir=_config.log_dir,
    )

    logger.info(
        "启动 IndexTTS2 TTS 引擎: host=%s, port=%d, device=%s",
        _config.host, _config.port, _config.device,
    )

    # 创建 app
    app = create_app(_config)

    # 用 uvicorn 启动
    import uvicorn
    uvicorn.run(
        app,
        host=_config.host,
        port=_config.port,
        log_level=_config.log_level.lower(),
    )


if __name__ == "__main__":
    main()
