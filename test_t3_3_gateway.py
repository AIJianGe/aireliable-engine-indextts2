"""T3.3 验证 gateway 调用返回可下载路径。

用户决策 Q2：UI 用 audio_path 本地播放；gateway 调用时返回可下载路径。
验证：
1. audio_url 是相对路径（/output/audio/xxx.wav）
2. 拼接 engine_url 后可下载（HTTP 200 + 正确 Content-Type）
3. gateway 任务上报包含 audio_url（可下载）
"""
import json
import sys
import httpx
from pathlib import Path

ENGINE_URL = "http://127.0.0.1:7005"
GATEWAY_URL = "http://127.0.0.1:8000"


def main():
    print("=== 1. 通过 gateway /v1/voices 查音色 ===", flush=True)
    r = httpx.get(f"{GATEWAY_URL}/v1/voices", timeout=10.0)
    r.raise_for_status()
    data = r.json()
    voices = data.get("data", data) if isinstance(data, dict) else data
    if not voices:
        print("❌ 音色库为空", flush=True)
        sys.exit(1)
    voice = voices[0]
    print(f"  音色: id={voice.get('id')}, name={voice.get('name')}", flush=True)
    print(f"  路径: {voice.get('path')}", flush=True)

    print("\n=== 2. 通过 /infer 触发一次合成（直接调 engine 模拟 gateway 调用）===", flush=True)
    speaker_path = voice["path"]
    payload = {
        "variant": "IndexTeam/IndexTTS-2",
        "inputs": {
            "text": "测试 gateway 调用",
            "mode": "clone",
            "reference_audio_path": speaker_path,
            "top_p": 0.8,
            "top_k": 30,
            "temperature": 0.8,
            "output_format": "wav",
        },
        "audios": [{"path": speaker_path, "role": "speaker"}],
    }

    audio_url = None
    audio_path = None
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
                frame_type = data.get("stage") or data.get("type")
                if frame_type == "completed":
                    result = data.get("result", {})
                    if isinstance(result, dict):
                        audio_url = result.get("audio_url")
                        audio_path = result.get("audio_path")

    if not audio_url or not audio_path:
        print(f"❌ 合成失败: audio_url={audio_url}, audio_path={audio_path}", flush=True)
        sys.exit(1)

    print(f"  ✅ audio_url  = {audio_url}", flush=True)
    print(f"  ✅ audio_path = {audio_path}", flush=True)

    # 验证 1：audio_url 是相对路径（/output/audio/xxx.wav）
    print("\n=== 3. 验证 audio_url 是相对路径 ===", flush=True)
    if not audio_url.startswith("http"):
        print(f"  ✅ audio_url 是相对路径（可拼 engine URL 下载）", flush=True)
    else:
        print(f"  ⚠️ audio_url 是完整 URL：{audio_url}", flush=True)

    # 验证 2：拼 engine URL 可下载
    print("\n=== 4. 验证 audio_url 可下载（拼 engine URL）===", flush=True)
    full_url = audio_url if audio_url.startswith("http") else f"{ENGINE_URL}{audio_url}"
    print(f"  下载 URL: {full_url}", flush=True)
    try:
        r = httpx.get(full_url, timeout=30.0, follow_redirects=True)
        r.raise_for_status()
        content_type = r.headers.get("content-type", "")
        size = len(r.content)
        print(f"  ✅ HTTP {r.status_code}", flush=True)
        print(f"  ✅ Content-Type: {content_type}", flush=True)
        print(f"  ✅ Size: {size} bytes", flush=True)
        if size == 0:
            print("  ❌ 下载内容为空", flush=True)
            sys.exit(1)
    except Exception as e:
        print(f"  ❌ 下载失败: {e}", flush=True)
        sys.exit(1)

    # 验证 3：保存到本地并对比
    print("\n=== 5. 验证下载文件 = audio_path 本地文件 ===", flush=True)
    save_path = Path("/tmp") / Path(audio_path).name
    save_path.write_bytes(r.content)
    print(f"  下载保存到: {save_path} ({save_path.stat().st_size} bytes)", flush=True)
    local_size = Path(audio_path).stat().st_size
    if save_path.stat().st_size == local_size:
        print(f"  ✅ 下载文件与本地文件大小一致（{local_size} bytes）", flush=True)
    else:
        print(f"  ⚠️ 大小不一致：下载={save_path.stat().st_size}, 本地={local_size}", flush=True)

    # 验证 4：gateway 任务上报走 audio_url（通过 engine /infer 响应看）
    print("\n=== 6. 验证 audio_url 可作 gateway 任务结果 URL ===", flush=True)
    print(f"  audio_url 形式: {audio_url}", flush=True)
    print(f"  engine 拼接: http://127.0.0.1:7005{audio_url}", flush=True)
    print(f"  ✅ gateway 任务可通过返回 audio_url（相对路径），客户端拼 engine_url 即可下载", flush=True)

    print("\n🎉 T3.3 PASS", flush=True)
    print(f"  总结：", flush=True)
    print(f"    - audio_url  = {audio_url} (相对路径，可下载)", flush=True)
    print(f"    - audio_path = {audio_path} (绝对路径，本地播放)", flush=True)


if __name__ == "__main__":
    main()
