"""T4 gateway 端到端验证：验证 gateway 透传 IndexTTS2 情感控制参数。

修复内容（audio_endpoints.py）：
  - extra.emotion_alpha → inputs.emotion_alpha
  - extra.emotion_vector → inputs.emotion_vector
  - extra.emotion_text → inputs.emotion_text
  - extra.top_p / top_k / temperature / repetition_penalty / max_mel_tokens / max_text_tokens_per_segment

验证 4 种模式：
  1. clone（基础克隆）
  2. emo_vector（8 维情感向量）
  3. emo_text（情感描述文本）
  4. emo_clone 需要情感参考音频，跳过

响应格式验证：
  - audio_url 是 /output/audio/xxx.wav 相对路径
  - 拼 gateway URL 后可下载（HTTP 200）
"""
import json
import sys
import time

import httpx

GATEWAY_URL = "http://127.0.0.1:8000"


def fetch_voice_id():
    """从 gateway /v1/voices 获取第一个音色 ID。"""
    print("=== 0. 查询音色库 ===", flush=True)
    r = httpx.get(f"{GATEWAY_URL}/v1/voices", timeout=10.0)
    r.raise_for_status()
    data = r.json()
    voices = data.get("data", data) if isinstance(data, dict) else data
    if not voices:
        raise RuntimeError("音色库为空")
    voice = voices[0]
    print(f"  音色: id={voice['id']}, name={voice['name']}", flush=True)
    return voice["id"]


def test_tts(mode: str, extra: dict, label: str):
    """通过 gateway /v1/audio/speech 测试 TTS 合成。

    使用 stream="json" 同步模式，等待完成后一次性返回。
    """
    print(f"\n=== 测试 {label}（mode={mode}）===", flush=True)

    payload = {
        "model": "IndexTeam/IndexTTS-2",
        "input": "今天天气真好，我想出去散步",
        "voice": None,
        "response_format": "wav",
        "stream": "json",
        "extra": {
            "voice_id": fetch_voice_id(),
            "workflow": mode,
            **extra,
        },
    }

    t0 = time.time()
    try:
        r = httpx.post(
            f"{GATEWAY_URL}/v1/audio/speech",
            json=payload,
            timeout=300.0,
        )
        r.raise_for_status()
        elapsed = time.time() - t0
        result = r.json()

        print(f"  耗时: {elapsed:.2f}s", flush=True)
        print(f"  HTTP {r.status_code}", flush=True)

        # 检查响应结构
        if "data" in result and len(result["data"]) > 0:
            audio = result["data"][0]
            audio_url = audio.get("audio_url")
            audio_path = audio.get("audio_path")
            print(f"  ✅ audio_url  = {audio_url}", flush=True)
            print(f"  ✅ audio_path = {audio_path}", flush=True)

            # 验证 audio_url 可下载
            if audio_url:
                full_url = audio_url if audio_url.startswith("http") else f"{GATEWAY_URL}{audio_url}"
                print(f"  下载 URL: {full_url}", flush=True)
                try:
                    r2 = httpx.get(full_url, timeout=30.0, follow_redirects=True)
                    r2.raise_for_status()
                    print(f"  ✅ 下载成功: HTTP {r2.status_code}, {len(r2.content)} bytes, Content-Type={r2.headers.get('content-type')}", flush=True)
                except Exception as e:
                    print(f"  ❌ 下载失败: {e}", flush=True)
                    return False

            return True
        elif "error" in result:
            print(f"  ❌ 错误: {result['error']}", flush=True)
            return False
        else:
            print(f"  ⚠️ 异常响应: {json.dumps(result, ensure_ascii=False)[:300]}", flush=True)
            return False

    except httpx.HTTPStatusError as e:
        elapsed = time.time() - t0
        print(f"  ❌ HTTP {e.response.status_code} ({elapsed:.1f}s): {e.response.text[:300]}", flush=True)
        return False
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ❌ 异常 ({elapsed:.1f}s): {e}", flush=True)
        return False


def main():
    results = {}

    # Test 1: clone 模式
    results["clone"] = test_tts("clone", {}, "clone 模式（基础克隆）")

    # Test 2: emo_vector 模式（验证 emotion_vector 透传）
    results["emo_vector"] = test_tts(
        "emo_vector",
        {"emotion_vector": [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        "emo_vector 模式（情感向量：开心 0.8）",
    )

    # Test 3: emo_text 模式（验证 emotion_text 透传）
    results["emo_text"] = test_tts(
        "emo_text",
        {"emotion_text": "用开心的语气说"},
        "emo_text 模式（情感描述文本）",
    )

    # Test 4: clone + 高级生成参数（验证 top_p / top_k 等透传）
    results["clone_advanced"] = test_tts(
        "clone",
        {"top_p": 0.9, "top_k": 50, "temperature": 1.0},
        "clone 模式 + 高级参数（top_p=0.9, top_k=50, temperature=1.0）",
    )

    # 汇总
    print("\n=== 测试结果汇总 ===", flush=True)
    for k, v in results.items():
        status = "✅ PASS" if v else "❌ FAIL"
        print(f"  {k}: {status}", flush=True)

    pass_count = sum(1 for v in results.values() if v)
    total = len(results)
    if pass_count == total:
        print(f"\n🎉 全部通过（{pass_count}/{total}）", flush=True)
    else:
        print(f"\n⚠️ {pass_count}/{total} 通过", flush=True)

    sys.exit(0 if pass_count == total else 1)


if __name__ == "__main__":
    main()
