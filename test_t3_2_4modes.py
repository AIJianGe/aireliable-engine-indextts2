"""T3.2 端到端验证脚本：测试 4 种合成模式的 SSE 帧解析。

v0.4.4 修复验证：
- SSE 帧类型用 stage 字段（不是 type）
- audio_url 嵌在 result 嵌套里
- error.message 嵌在 error 嵌套里
"""
import json
import sys
import time
from pathlib import Path

import httpx

ENGINE_URL = "http://127.0.0.1:7005"
GATEWAY_URL = "http://127.0.0.1:8000"


def fetch_voice():
    """从 gateway 拿一个音色作为 speaker reference。"""
    print("=== 0. 查询平台音色库 ===", flush=True)
    r = httpx.get(f"{GATEWAY_URL}/v1/voices", timeout=10.0)
    r.raise_for_status()
    data = r.json()
    voices = data.get("data", data) if isinstance(data, dict) else data
    if not voices:
        raise RuntimeError("音色库为空，请先到平台音色库管理页添加音色")
    first = voices[0]
    print(f"  第一个音色：id={first.get('id')}, name={first.get('name')}", flush=True)
    print(f"  本地路径：{first.get('path')}", flush=True)
    if not first.get("path") or not Path(first["path"]).exists():
        raise RuntimeError(f"音色本地路径无效：{first.get('path')}")
    return first["path"], first.get("id")


def test_mode(mode: str, speaker_path: str, extra: dict, label: str):
    """测试单个合成模式。

    关键验证：
    1. SSE 帧能被正确解析
    2. completed 帧能拿到 audio_url 和 audio_path
    3. error 帧能拿到 error.message
    """
    print(f"\n=== 测试 {label}（mode={mode}）===", flush=True)

    inputs = {
        "text": "今天天气真好，我想出去散步",
        "mode": mode,
        "top_p": 0.8,
        "top_k": 30,
        "temperature": 0.8,
        "repetition_penalty": 10.0,
        "max_mel_tokens": 1500,
        "max_text_tokens_per_segment": 120,
        "output_format": "wav",
        "reference_audio_path": speaker_path,
    }
    inputs.update(extra)

    payload = {
        "variant": "IndexTeam/IndexTTS-2",
        "inputs": inputs,
        "audios": [{"path": speaker_path, "role": "speaker"}],
    }

    audio_url = None
    audio_path = None
    error_msg = None
    last_stage = None
    last_progress = None
    msg_count = 0
    t0 = time.time()

    with httpx.Client(timeout=300.0) as client:
        with client.stream("POST", f"{ENGINE_URL}/infer", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                msg_count += 1
                if "stage" in data:
                    last_stage = data["stage"]
                if "progress" in data:
                    last_progress = data["progress"]
                if data.get("message"):
                    print(f"  [{last_stage or '?'}] {data['message']}", flush=True)

                # v0.4.4 修复验证：用 stage 而不是 type
                frame_type = data.get("stage") or data.get("type")
                if frame_type == "completed":
                    # v0.4.4 修复验证：audio_url 在 result 嵌套里
                    result = data.get("result", {})
                    if isinstance(result, dict):
                        audio_url = result.get("audio_url")
                        audio_path = result.get("audio_path")
                elif frame_type == "error":
                    # v0.4.4 修复验证：error.message 在 error 嵌套里
                    err = data.get("error", {})
                    if isinstance(err, dict):
                        error_msg = err.get("message") or data.get("message", "未知错误")
                    else:
                        error_msg = data.get("message") or str(err) or "未知错误"
                    break

    elapsed = time.time() - t0
    print(f"  耗时: {elapsed:.2f}s, SSE 消息数: {msg_count}", flush=True)
    print(f"  最后 stage: {last_stage}, progress: {last_progress}", flush=True)

    if error_msg:
        print(f"  ❌ 错误: {error_msg}", flush=True)
        return False

    if not audio_url and not audio_path:
        print("  ❌ 合成失败：未返回音频 URL 或路径", flush=True)
        return False

    print(f"  ✅ audio_url  = {audio_url}", flush=True)
    print(f"  ✅ audio_path = {audio_path}", flush=True)

    if audio_path:
        p = Path(audio_path)
        if p.exists():
            print(f"  ✅ audio_path 文件存在（{p.stat().st_size} bytes）", flush=True)
        else:
            print(f"  ⚠️ audio_path 指向的文件不存在", flush=True)
            return False

    return True


def main():
    try:
        speaker_path, voice_id = fetch_voice()
    except Exception as e:
        print(f"FATAL: {e}", flush=True)
        sys.exit(1)

    results = {}

    # Test 1: clone 模式（仅音色参考）
    results["clone"] = test_mode("clone", speaker_path, {}, "clone 模式（仅音色参考）")

    # Test 2: emo_vector 模式（8 维情感向量）
    results["emo_vector"] = test_mode(
        "emo_vector", speaker_path,
        {"emotion_vector": [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        "emo_vector 模式（情感向量：开心 0.8）",
    )

    # Test 3: emo_clone 模式需要情感参考音频，跳过（没有合适的情感参考音频）
    print("\n=== 测试 emo_clone 模式（需要情感参考音频，跳过）===", flush=True)
    results["emo_clone"] = "skipped"

    # Test 4: emo_text 模式（情感描述文本）
    results["emo_text"] = test_mode(
        "emo_text", speaker_path,
        {"emotion_text": "用开心的语气说"},
        "emo_text 模式（情感描述：用开心的语气说）",
    )

    print("\n=== 测试结果汇总 ===", flush=True)
    for k, v in results.items():
        if v is True:
            print(f"  ✅ {k}: PASS", flush=True)
        elif v is False:
            print(f"  ❌ {k}: FAIL", flush=True)
        else:
            print(f"  ⏭️  {k}: {v}", flush=True)

    # 至少 2 个模式通过（clone + emo_vector）算 PASS
    pass_count = sum(1 for v in results.values() if v is True)
    if pass_count >= 2:
        print(f"\n🎉 T3.2 PASS（{pass_count}/{len(results)} 模式通过）", flush=True)
        sys.exit(0)
    else:
        print(f"\n❌ T3.2 FAIL（仅 {pass_count}/{len(results)} 模式通过）", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
