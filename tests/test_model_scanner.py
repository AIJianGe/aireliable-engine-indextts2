"""IndexTTS2 ModelScanner 单元测试（P6-2）。

测试关键场景：
    1. 平台 API 路径（mock engine.get_variants 返回非 None）
    2. 本地 fallback 路径（engine=None）
    3. 标志文件缺失/存在场景
    4. resolve_variant_path

运行（需项目根目录在 sys.path）::
    cd envs/voice-voxcpm
    .venv/Scripts/python.exe -m unittest engines.indextts2_b7c3d9e1.tests.test_model_scanner -v
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# 确保项目根目录在 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from engines.indextts2_b7c3d9e1.model_scanner import ModelScanner, _KNOWN_VARIANTS


class TestModelScannerLocal(unittest.TestCase):
    """本地 fallback 扫描测试（engine=None）。"""

    def setUp(self):
        self._tmpdirs = []

    def _make_dir(self):
        d = tempfile.mkdtemp()
        self._tmpdirs.append(d)
        return Path(d)

    def test_scan_empty_dir(self):
        """空目录 → 返回空列表。"""
        model_dir = self._make_dir()
        scanner = ModelScanner(model_dir, engine=None)
        result = scanner.scan_variants()
        self.assertEqual(result, [])

    def test_scan_with_config_yaml(self):
        """存在 config.yaml 标志文件 → 识别为可用 variant。"""
        model_dir = self._make_dir()
        v_dir = model_dir / "IndexTTS-2"
        v_dir.mkdir(parents=True)
        (v_dir / "config.yaml").write_text("dummy", encoding="utf-8")
        scanner = ModelScanner(model_dir, engine=None)
        result = scanner.scan_variants()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "IndexTeam/IndexTTS-2")
        self.assertEqual(result[0].alias, "IndexTTS-2")
        self.assertEqual(result[0].sample_rate, 24000)
        self.assertIn("clone", result[0].modes)

    def test_scan_without_config_yaml(self):
        """目录存在但无标志文件 → 不识别。"""
        model_dir = self._make_dir()
        v_dir = model_dir / "IndexTTS-2"
        v_dir.mkdir(parents=True)
        (v_dir / "random.txt").write_text("not a marker", encoding="utf-8")
        scanner = ModelScanner(model_dir, engine=None)
        result = scanner.scan_variants()
        self.assertEqual(result, [])

    def test_scan_nonexistent_dir(self):
        """模型根目录不存在 → 返回空列表，不抛异常。"""
        model_dir = self._make_dir() / "nonexistent"
        scanner = ModelScanner(model_dir, engine=None)
        result = scanner.scan_variants()
        self.assertEqual(result, [])

    def test_resolve_variant_path_exists(self):
        """resolve_variant_path 返回存在的路径。"""
        model_dir = self._make_dir()
        v_dir = model_dir / "IndexTTS-2"
        v_dir.mkdir(parents=True)
        (v_dir / "config.yaml").write_text("dummy", encoding="utf-8")
        scanner = ModelScanner(model_dir, engine=None)
        path = scanner.resolve_variant_path("IndexTeam/IndexTTS-2")
        self.assertIsNotNone(path)
        self.assertTrue(path.exists())

    def test_resolve_variant_path_not_found(self):
        """resolve_variant_path 未知 variant → 返回 None。"""
        model_dir = self._make_dir()
        scanner = ModelScanner(model_dir, engine=None)
        path = scanner.resolve_variant_path("Unknown/Model")
        self.assertIsNone(path)


class TestModelScannerPlatformAPI(unittest.TestCase):
    """平台 API 路径测试（mock engine.get_variants）。"""

    def _make_dir(self):
        d = tempfile.mkdtemp()
        return Path(d)

    def test_platform_api_returns_variants(self):
        """平台 API 返回非 None → 使用平台数据。"""
        model_dir = self._make_dir()
        mock_engine = MagicMock()
        mock_engine.get_variants = AsyncMock(return_value=[
            {"name": "IndexTeam/IndexTTS-2", "alias": "IndexTTS-2", "available": True},
        ])
        scanner = ModelScanner(model_dir, engine=mock_engine)
        result = asyncio.run(scanner.scan_variants_async())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "IndexTeam/IndexTTS-2")
        # 引擎特有字段（sample_rate）来自 _KNOWN_VARIANTS
        self.assertEqual(result[0].sample_rate, 24000)

    def test_platform_api_returns_none_fallback(self):
        """平台 API 返回 None → fallback 到本地扫描。"""
        model_dir = self._make_dir()
        v_dir = model_dir / "IndexTTS-2"
        v_dir.mkdir(parents=True)
        (v_dir / "config.yaml").write_text("dummy", encoding="utf-8")
        mock_engine = MagicMock()
        mock_engine.get_variants = AsyncMock(return_value=None)
        scanner = ModelScanner(model_dir, engine=mock_engine)
        result = asyncio.run(scanner.scan_variants_async())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "IndexTeam/IndexTTS-2")

    def test_platform_api_exception_fallback(self):
        """平台 API 异常 → fallback 到本地扫描。"""
        model_dir = self._make_dir()
        v_dir = model_dir / "IndexTTS-2"
        v_dir.mkdir(parents=True)
        (v_dir / "config.yaml").write_text("dummy", encoding="utf-8")
        mock_engine = MagicMock()
        mock_engine.get_variants = AsyncMock(side_effect=ConnectionError("network error"))
        scanner = ModelScanner(model_dir, engine=mock_engine)
        result = asyncio.run(scanner.scan_variants_async())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "IndexTeam/IndexTTS-2")

    def test_platform_api_empty_list(self):
        """平台 API 返回空列表 → 返回空（平台说无可用 variant）。"""
        model_dir = self._make_dir()
        mock_engine = MagicMock()
        mock_engine.get_variants = AsyncMock(return_value=[])
        scanner = ModelScanner(model_dir, engine=mock_engine)
        result = asyncio.run(scanner.scan_variants_async())
        self.assertEqual(result, [])


class TestKnownVariants(unittest.TestCase):
    """_KNOWN_VARIANTS 常量验证。"""

    def test_known_variants_format(self):
        """variant name 必须为 author/model 格式。"""
        for v in _KNOWN_VARIANTS:
            self.assertIn("/", v["name"], f"variant name 必须为 author/model 格式: {v['name']}")

    def test_known_variants_fields(self):
        """每个 variant 定义包含必需字段。"""
        required_fields = {"name", "alias", "precision", "vram_required",
                          "sample_rate", "languages", "subdir", "modes", "emotion_labels"}
        for v in _KNOWN_VARIANTS:
            missing = required_fields - set(v.keys())
            self.assertFalse(missing, f"variant {v['name']} 缺少字段: {missing}")

    def test_known_variants_modes_v01(self):
        """v0.1 仅支持 clone/emo_clone/emo_vector 三种模式。"""
        for v in _KNOWN_VARIANTS:
            self.assertEqual(v["modes"], ["clone", "emo_clone", "emo_vector"])


if __name__ == "__main__":
    unittest.main()
