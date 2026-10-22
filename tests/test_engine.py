"""IndexTTS2 引擎单元测试（P6-1）。

测试关键场景：
    1. IndexTTS2Inputs 字段验证（Pydantic 模型）
    2. manifest.yaml 一致性检查（model_file_ids 与 model_files 对应）
    3. _read_default_variant_from_manifest（mock 引擎实例）
    4. IndexTTS2Capability 字段验证

运行（需项目根目录在 sys.path，需 aireliable_engine SDK）::
    cd D:\myProInstallDir\AIReliable
    envs/voice-voxcpm/.venv/Scripts/python.exe -m unittest engines.indextts2_b7c3d9e1.tests.test_engine -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# 确保项目根目录在 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from engines.indextts2_b7c3d9e1.schemas import (
    EMOTION_LABELS,
    INDEXTTS2_MODES,
    SAMPLE_RATE,
    IndexTTS2Capability,
    IndexTTS2Inputs,
    IndexTTS2StatusExtra,
)


# ===========================================================================
# IndexTTS2Inputs 字段验证
# ===========================================================================

class TestIndexTTS2Inputs(unittest.TestCase):
    """IndexTTS2Inputs Pydantic 模型验证。"""

    def test_minimal_valid(self):
        """最小有效输入：仅 text。"""
        inputs = IndexTTS2Inputs(text="你好世界")
        self.assertEqual(inputs.text, "你好世界")
        self.assertEqual(inputs.mode, "clone")
        self.assertEqual(inputs.emotion_alpha, 1.0)
        self.assertEqual(inputs.top_p, 0.8)
        self.assertEqual(inputs.max_mel_tokens, 1500)

    def test_text_required(self):
        """text 为必填字段。"""
        with self.assertRaises(Exception):
            IndexTTS2Inputs()

    def test_emotion_vector_length_8(self):
        """emotion_vector 必须为 8 维。"""
        # 7 维 → 报错
        with self.assertRaises(Exception):
            IndexTTS2Inputs(text="test", emotion_vector=[0.1] * 7)
        # 9 维 → 报错
        with self.assertRaises(Exception):
            IndexTTS2Inputs(text="test", emotion_vector=[0.1] * 9)
        # 8 维 → OK
        inputs = IndexTTS2Inputs(text="test", emotion_vector=[0.1] * 8)
        self.assertEqual(len(inputs.emotion_vector), 8)

    def test_emotion_alpha_range(self):
        """emotion_alpha 范围 0.0-1.0。"""
        IndexTTS2Inputs(text="test", emotion_alpha=0.0)
        IndexTTS2Inputs(text="test", emotion_alpha=1.0)
        with self.assertRaises(Exception):
            IndexTTS2Inputs(text="test", emotion_alpha=-0.1)
        with self.assertRaises(Exception):
            IndexTTS2Inputs(text="test", emotion_alpha=1.1)

    def test_mode_default(self):
        """mode 默认值为 clone。"""
        inputs = IndexTTS2Inputs(text="test")
        self.assertEqual(inputs.mode, "clone")

    def test_generation_params_defaults(self):
        """生成参数默认值与 IndexTTS2 webui.py 预设一致。"""
        inputs = IndexTTS2Inputs(text="test")
        self.assertTrue(inputs.do_sample)
        self.assertEqual(inputs.top_p, 0.8)
        self.assertEqual(inputs.top_k, 30)
        self.assertEqual(inputs.temperature, 0.8)
        self.assertEqual(inputs.repetition_penalty, 10.0)
        self.assertEqual(inputs.max_mel_tokens, 1500)
        self.assertEqual(inputs.max_text_tokens_per_segment, 120)

    def test_output_format_default(self):
        """output_format 默认 wav。"""
        inputs = IndexTTS2Inputs(text="test")
        self.assertEqual(inputs.output_format, "wav")


# ===========================================================================
# IndexTTS2Capability 字段验证
# ===========================================================================

class TestIndexTTS2Capability(unittest.TestCase):
    """IndexTTS2Capability 模型验证。"""

    def test_default_fields(self):
        """默认字段值正确。"""
        cap = IndexTTS2Capability(name="IndexTeam/IndexTTS-2", alias="IndexTTS-2")
        self.assertEqual(cap.name, "IndexTeam/IndexTTS-2")
        self.assertEqual(cap.precision, "fp16")
        self.assertEqual(cap.sample_rate, SAMPLE_RATE)
        self.assertEqual(cap.sample_rate, 24000)
        self.assertIn("zh", cap.languages)
        self.assertIn("en", cap.languages)
        # v0.1 仅 clone/emo_clone/emo_vector
        self.assertEqual(cap.modes, ["clone", "emo_clone", "emo_vector"])
        self.assertEqual(len(cap.emotion_labels), 8)
        self.assertFalse(cap.loaded)

    def test_name_required(self):
        """name 和 alias 为必填。"""
        with self.assertRaises(Exception):
            IndexTTS2Capability()


# ===========================================================================
# IndexTTS2StatusExtra 字段验证
# ===========================================================================

class TestIndexTTS2StatusExtra(unittest.TestCase):
    """IndexTTS2StatusExtra 模型验证。"""

    def test_defaults(self):
        """默认字段值正确。"""
        extra = IndexTTS2StatusExtra()
        self.assertEqual(extra.device, "cpu")
        self.assertEqual(extra.sample_rate, SAMPLE_RATE)
        self.assertFalse(extra.is_loaded)
        self.assertIsNone(extra.loaded_variant)


# ===========================================================================
# 常量验证
# ===========================================================================

class TestConstants(unittest.TestCase):
    """模块常量验证。"""

    def test_modes(self):
        """INDEXTTS2_MODES 包含 4 种模式。"""
        self.assertEqual(len(INDEXTTS2_MODES), 4)
        self.assertIn("clone", INDEXTTS2_MODES)
        self.assertIn("emo_clone", INDEXTTS2_MODES)
        self.assertIn("emo_vector", INDEXTTS2_MODES)
        self.assertIn("emo_text", INDEXTTS2_MODES)

    def test_emotion_labels_count(self):
        """EMOTION_LABELS 为 8 维。"""
        self.assertEqual(len(EMOTION_LABELS), 8)

    def test_sample_rate(self):
        """SAMPLE_RATE 为 24000。"""
        self.assertEqual(SAMPLE_RATE, 24000)


# ===========================================================================
# manifest.yaml 一致性检查
# ===========================================================================

class TestManifestConsistency(unittest.TestCase):
    """manifest.yaml 一致性检查（P1-2-4 / P1-2-5 验证）。"""

    @classmethod
    def setUpClass(cls):
        """读取 manifest.yaml。"""
        import yaml
        manifest_path = _PROJECT_ROOT / "engines" / "indextts2_b7c3d9e1" / "manifest.yaml"
        with open(manifest_path, encoding="utf-8") as f:
            cls.manifest = yaml.safe_load(f)

    def test_manifest_has_models_section(self):
        """manifest 包含 models 部分。"""
        self.assertIn("models", self.manifest)

    def test_model_file_ids_match_model_files(self):
        """variants[0].model_file_ids 与 model_files 列表完全一致。"""
        models = self.manifest["models"]
        mf = models["model_files"]
        ids = [f["id"] for f in mf]
        variant = models["variants"][0]
        self.assertEqual(set(variant["model_file_ids"]), set(ids),
                         "model_file_ids 与 model_files 不一致")

    def test_variant_name_author_model_format(self):
        """variant.name 为 author/model 格式。"""
        for v in self.manifest["models"]["variants"]:
            self.assertIn("/", v["name"], f"variant name 必须为 author/model 格式: {v['name']}")

    def test_manifest_has_engine_id(self):
        """manifest 包含 engine_id。"""
        self.assertIn("engine_id", self.manifest)
        self.assertEqual(self.manifest["engine_id"], "indextts2_b7c3d9e1")

    def test_engine_config_has_port(self):
        """config/engines/ 配置文件包含端口 7005。"""
        import yaml
        cfg_path = _PROJECT_ROOT / "config" / "engines" / "indextts2_b7c3d9e1.yaml"
        self.assertTrue(cfg_path.exists(), "config/engines/indextts2_b7c3d9e1.yaml 不存在")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        deploy = cfg.get("deployment", {})
        local_cfg = deploy.get("local_config", {})
        port = local_cfg.get("port")
        self.assertEqual(port, 7005, f"端口应为 7005，实际 {port}")

    def test_model_files_have_required_fields(self):
        """每个 model_file 包含必需字段。"""
        required = {"id", "filename", "target_dir", "format"}
        for mf in self.manifest["models"]["model_files"]:
            missing = required - set(mf.keys())
            self.assertFalse(missing, f"model_file {mf.get('id')} 缺少字段: {missing}")

    def test_marker_files_have_key_files(self):
        """format=marker 的 model_file 必须声明 key_files。"""
        for mf in self.manifest["models"]["model_files"]:
            if mf.get("format") == "marker":
                self.assertIn("key_files", mf, f"marker 文件 {mf['id']} 缺少 key_files")
                self.assertTrue(mf["key_files"], f"marker 文件 {mf['id']} 的 key_files 为空")


# ===========================================================================
# _read_default_variant_from_manifest 测试
# ===========================================================================

class TestReadDefaultVariant(unittest.TestCase):
    """_read_default_variant_from_manifest 方法测试。

    通过 mock 引擎实例（绕过 CustomEngine.__init__）测试此方法。
    """

    def _make_mock_engine(self):
        """创建 mock 引擎实例（绕过 __init__）。"""
        try:
            from engines.indextts2_b7c3d9e1.engine import IndexTTS2Engine
            # __new__ 绕过 __init__
            return IndexTTS2Engine.__new__(IndexTTS2Engine)
        except Exception:
            self.skipTest("无法导入 IndexTTS2Engine（SDK 未安装？）")

    def test_read_default_variant_from_temp_manifest(self):
        """从临时 manifest 读取 default_variant。"""
        import yaml
        engine = self._make_mock_engine()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump({
                "models": {
                    "default_variant": "IndexTeam/IndexTTS-2",
                    "variants": [{"name": "IndexTeam/IndexTTS-2"}],
                }
            }, f)
            manifest_path = Path(f.name)

        try:
            result = engine._read_default_variant_from_manifest(manifest_path)
            self.assertEqual(result, "IndexTeam/IndexTTS-2")
        finally:
            manifest_path.unlink(missing_ok=True)

    def test_read_default_variant_missing_file(self):
        """manifest 文件不存在 → 返回 None。"""
        engine = self._make_mock_engine()
        result = engine._read_default_variant_from_manifest(Path("nonexistent.yaml"))
        self.assertIsNone(result)

    def test_read_default_variant_no_default_field(self):
        """manifest 无 default_variant 字段 → 返回 None。"""
        import yaml
        engine = self._make_mock_engine()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump({"models": {"variants": []}}, f)
            manifest_path = Path(f.name)

        try:
            result = engine._read_default_variant_from_manifest(manifest_path)
            self.assertIsNone(result)
        finally:
            manifest_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
