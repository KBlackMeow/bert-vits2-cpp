"""Japanese HTTP API integration test for Bert-VITS2 C++ server."""

import json
import os
import struct
import sys
import urllib.error
import urllib.request

HOST = os.environ.get("BERT_VITS2_HOST", "127.0.0.1")
PORT = int(os.environ.get("BERT_VITS2_PORT", "7860"))
BASE_URL = f"http://{HOST}:{PORT}"

JP_REQUEST = {
    "text": "こんにちは、世界。今日はいい天気ですね。",
    "language": "JP",
    "speaker_name": "tachibana_ja",
    "length_scale": 1.0,
}

JP_STREAM_REQUEST = {
    "text": "こんにちは。これは日本語のストリーミングテストです。",
    "language": "JP",
    "speaker_name": "tachibana_ja",
    "stream": True,
}


def default_server_cmd() -> str:
    if sys.platform == "win32":
        return ".\\bin\\windows\\bert-vits2-project.exe --server --host 127.0.0.1 --port 7860"
    if sys.platform == "darwin":
        return "./bin/macos/bert-vits2-project --server --host 127.0.0.1 --port 7860"
    return "./bin/linux/bert-vits2-project --server --host 127.0.0.1 --port 7860"


    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        headers = {k.lower(): v for k, v in resp.headers.items()}
        return resp.status, resp.read(), headers


def assert_wav(payload: bytes) -> int:
    if len(payload) < 44:
        raise AssertionError(f"WAV too short: {len(payload)} bytes")
    if payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise AssertionError("response is not a WAV file")
    sample_rate = struct.unpack_from("<I", payload, 24)[0]
    if sample_rate != 44100:
        raise AssertionError(f"unexpected sample rate: {sample_rate}")
    data_size = struct.unpack_from("<I", payload, 40)[0]
    if data_size < 2:
        raise AssertionError(f"WAV data chunk too small: {data_size}")
    return sample_rate


def assert_pcm_stream(payload: bytes, headers: dict) -> int:
    content_type = headers.get("content-type", "")
    if "l16" not in content_type.lower():
        raise AssertionError(f"unexpected content-type: {content_type}")
    sample_rate = int(headers.get("x-sample-rate", "44100"))
    if sample_rate != 44100:
        raise AssertionError(f"unexpected X-Sample-Rate: {sample_rate}")
    if len(payload) < 2:
        raise AssertionError("PCM stream is empty")
    if len(payload) % 2 != 0:
        raise AssertionError("PCM byte length is not even")
    return sample_rate


def test_jp_tts_wav() -> None:
    status, body, _ = post_json("/tts", JP_REQUEST)
    if status != 200:
        raise AssertionError(f"/tts returned HTTP {status}")
    sample_rate = assert_wav(body)
    duration_sec = (len(body) - 44) / 2 / sample_rate
    if duration_sec < 0.5:
        raise AssertionError(f"audio too short: {duration_sec:.2f}s")
    print(f"jp_tts_wav ok: bytes={len(body)} duration={duration_sec:.2f}s")


def test_jp_tts_stream() -> None:
    status, body, headers = post_json("/tts/stream", JP_STREAM_REQUEST)
    if status != 200:
        raise AssertionError(f"/tts/stream returned HTTP {status}")
    sample_rate = assert_pcm_stream(body, headers)
    duration_sec = len(body) / 2 / sample_rate
    if duration_sec < 0.5:
        raise AssertionError(f"stream too short: {duration_sec:.2f}s")
    print(f"jp_tts_stream ok: bytes={len(body)} duration={duration_sec:.2f}s")


def main() -> int:
    try:
        health = urllib.request.urlopen(f"{BASE_URL}/health", timeout=5)
        if health.status != 200:
            print(f"health check failed: HTTP {health.status}")
            return 1
    except urllib.error.URLError as exc:
        print(f"server not reachable at {BASE_URL}: {exc}")
        print(f"start server: {default_server_cmd()}")
        return 1

    test_jp_tts_wav()
    test_jp_tts_stream()
    print("all japanese api tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
