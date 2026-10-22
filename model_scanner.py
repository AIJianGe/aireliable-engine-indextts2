"""IndexTTS2 模型文件扫描器（v0.4.3+ 引入）。

扫描 ``models/voice/tts/indextts2/`` 目录，识别可用的 IndexTTS2 模型变体。

v0.2.11+ 方案 B 改造（参考 VoxCPM）：
    优先调用平台 API（``GET /engines/{id}/variants``）获取 variant 可用性，
    平台不可达时 fallback 到本地文件扫描。

本地 fallback 设计原则（project-rules.md §第三部分 §5）：
    - 只检查**标志文件**存在性（如 ``config.yaml``），判断 variant 目录是否创建过
    - 大权重文件的完整性检查交给平台 API（``key_files`` 机制）或 ``load()`` 兜底
    - 不能比平台 API 更严格，避免用户下载了一半时 variant 列表为空

规范：PR/development-guide.md §28.20A.5
参考：``engines/voxcpm_tts_c4a8f2e7/model_scanner.py``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .schemas import EMOTION_LABELS, INDEXTTS2_MODES, IndexTTS2Capability

logger = logging.getLogger(__name__)


# ===========================================================================
# 已知模型变体定义（fallback + 引擎特有字段源）
# ===========================================================================
# v0.3.6+ variant.name 必须为 author/model 格式（命名空间规范，防同名冲突）
# 与 manifest.yaml variants 列表保持一致（契约约束）

_KNOWN_VARIANTS: list[dict] = [
    {
        "name": "IndexTeam/IndexTTS-2",
        "alias": "IndexTTS-2",
        "precision": "fp16",
        "vram_required": 8192,           # MB（FP16 ≈ 6GB，预留 buffer）
        "sample_rate": 24000,            # 24kHz（IndexTTS2 统一）
        "languages": ["zh", "en"],
        "subdir": "IndexTTS-2",
        "modes": list(INDEXTTS2_MODES[:3]),  # v0.1 仅 clone/emo_clone/emo_vector
        "emotion_labels": list(EMOTION_LABELS),
    },
]


# ===========================================================================
# ModelScanner
# ===========================================================================

class ModelScanner:
    """IndexTTS2 模型目录扫描器。

    扫描 ``model_dir`` 下的子目录，识别已下载的 IndexTTS2 模型变体。

    v0.2.11+ 方案 B：优先调用平台 API，fallback 到本地扫描。
    """

    def __init__(self, model_dir: Path, engine=None):
        """初始化扫描器。

        Args:
            model_dir: 模型根目录（如 ``models/voice/tts/indextts2``）
            engine: 可选的 BaseEngine 实例（v0.2.11+ 方案 B 注入）。
                传入时优先调用平台 API 检测 variant；
                为 None 时走 fallback 本地扫描逻辑（兼容旧测试）。
        """
        self.model_dir = model_dir
        self.engine = engine

    async def scan_variants_async(self) -> list[IndexTTS2Capability]:
        """v0.2.11+ 方案 B：异步扫描，优先调用平台 API。

        调用链：
            1. ``engine.get_variants()`` → 平台 API（声明式检测）
            2. 平台返回 None → fallback 到 ``_scan_locally()``
            3. 合并平台 available 字段 + ``_KNOWN_VARIANTS`` 引擎特有字段
        """
        logger.info("variant 扫描路径：模型目录 %s", self.model_dir)

        if self.engine is not None:
            try:
                platform_variants = await self.engine.get_variants()
                if platform_variants is not None:
                    logger.info(
                        "variant 扫描路径：平台 API（AIRELIABLE_CORE_URL 已注入），"
                        "返回 %d 个 variant",
                        len(platform_variants),
                    )
                    return self._merge_with_known(platform_variants)
            except Exception as e:
                logger.warning(
                    "variant 扫描路径：本地 fallback（原因：平台 API 异常 %s）", e
                )
        else:
            logger.warning(
                "variant 扫描路径：本地 fallback（原因：engine 未注入，方案 B 未启用）"
            )

        return self._scan_locally()

    def scan_variants(self) -> list[IndexTTS2Capability]:
        """同步扫描接口（向后兼容）。

        v0.2.11+ 注：若需使用方案 B 平台 API，请调用 ``scan_variants_async()``。
        此同步方法走 fallback 本地扫描逻辑。
        """
        return self._scan_locally()

    def _merge_with_known(
        self, platform_variants: list[dict]
    ) -> list[IndexTTS2Capability]:
        """合并平台 variant 数据 + ``_KNOWN_VARIANTS`` 引擎特有字段。

        Args:
            platform_variants: 平台 API 返回的 variant 列表（含 available 字段）

        Returns:
            IndexTTS2Capability 列表（available 来自平台，
            sample_rate / languages / modes / emotion_labels 来自 _KNOWN_VARIANTS）
        """
        known_lookup: dict[str, dict] = {v["name"]: v for v in _KNOWN_VARIANTS}

        result: list[IndexTTS2Capability] = []
        for pv in platform_variants:
            v_name = pv.get("name", "")
            kv = known_lookup.get(v_name, {})
            result.append(IndexTTS2Capability(
                name=v_name,
                alias=pv.get("alias", kv.get("alias", "")),
                precision=pv.get("precision", kv.get("precision", "fp16")),
                vram_required=kv.get("vram_required", 0),
                sample_rate=kv.get("sample_rate", 24000),
                languages=kv.get("languages", []),
                modes=kv.get("modes", list(INDEXTTS2_MODES[:3])),
                emotion_labels=kv.get("emotion_labels", list(EMOTION_LABELS)),
            ))
            if pv.get("available"):
                logger.info("发现模型变体: %s (%s)", v_name, pv.get("alias", ""))

        if not any(pv.get("available") for pv in platform_variants):
            logger.warning(
                "未在 %s 中找到任何 IndexTTS2 模型，"
                "请下载模型到对应目录（如 models/voice/tts/indextts2/IndexTTS-2/）",
                self.model_dir,
            )

        return result

    def _scan_locally(self) -> list[IndexTTS2Capability]:
        """fallback 本地扫描（原 scan_variants 逻辑）。

        本地 fallback 设计原则（project-rules.md §第三部分 §5）：
            - 只检查标志文件（``config.yaml``）存在性
            - 大权重文件完整性检查交给平台 API 或 load() 兜底
            - 不能比平台 API 更严格
        """
        capabilities: list[IndexTTS2Capability] = []

        if not self.model_dir.exists():
            logger.warning("模型目录不存在: %s", self.model_dir)
            return capabilities

        for variant_def in _KNOWN_VARIANTS:
            variant_dir = self.model_dir / variant_def["subdir"]

            if not variant_dir.exists():
                logger.debug("变体 %s 目录不存在: %s", variant_def["name"], variant_dir)
                continue

            # 只检查标志文件存在性（不检查大权重文件，避免比平台 API 更严格）
            has_model = self._check_marker_files(variant_dir)
            if not has_model:
                logger.debug(
                    "变体 %s 目录内无标志文件 config.yaml: %s",
                    variant_def["name"], variant_dir,
                )
                continue

            cap = IndexTTS2Capability(
                name=variant_def["name"],
                alias=variant_def["alias"],
                precision=variant_def["precision"],
                vram_required=variant_def["vram_required"],
                sample_rate=variant_def["sample_rate"],
                languages=variant_def["languages"],
                modes=variant_def["modes"],
                emotion_labels=variant_def["emotion_labels"],
            )
            capabilities.append(cap)
            logger.info("发现模型变体: %s (%s)", cap.name, cap.alias)

        if not capabilities:
            logger.warning(
                "未在 %s 中找到任何 IndexTTS2 模型，"
                "请下载模型到对应目录（如 models/voice/tts/indextts2/IndexTTS-2/）",
                self.model_dir,
            )

        return capabilities

    def _check_marker_files(self, variant_dir: Path) -> bool:
        """检查目录内是否有标志文件（判断 variant 目录是否创建过）。

        IndexTTS2 模型为多文件格式，标志文件为 ``config.yaml``。
        本地 fallback 只检查标志文件，不检查大权重文件（如 gpt.pth ~1.5GB），
        避免用户下载了一半时 variant 列表为空，无法触发下载入口。

        Args:
            variant_dir: variant 目录路径

        Returns:
            True 表示目录已创建（标志文件存在）
        """
        # config.yaml 是 IndexTTS2 主模型配置文件（标志文件）
        if (variant_dir / "config.yaml").exists():
            return True

        # 兼容：检查权重文件（备用，确保迁移期间也能识别）
        for pattern in ("*.pth", "*.safetensors"):
            if list(variant_dir.glob(pattern)):
                return True

        return False

    def resolve_variant_path(self, variant_name: str) -> Optional[Path]:
        """根据变体名解析模型路径。

        Args:
            variant_name: 变体名（如 ``IndexTeam/IndexTTS-2``）

        Returns:
            模型目录路径，不存在返回 None
        """
        for variant_def in _KNOWN_VARIANTS:
            if variant_def["name"] == variant_name:
                path = self.model_dir / variant_def["subdir"]
                return path if path.exists() else None
        # 兼容：直接用 variant_name 作为子目录名
        direct_path = self.model_dir / variant_name.split("/")[-1]
        return direct_path if direct_path.exists() else None
