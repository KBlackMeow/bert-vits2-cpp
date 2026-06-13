#!/usr/bin/env python3
"""Stream TTS from bert-vits2-cpp HTTP API and play PCM while receiving."""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
import urllib.error
import urllib.request

try:
    import sounddevice as sd
except ImportError:
    sd = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream /tts/stream PCM and play audio while chunks arrive."
    )
    parser.add_argument("--host", default="127.0.0.1", help="server host")
    parser.add_argument("--port", type=int, default=7860, help="server port")
    parser.add_argument(
        "--path",
        default="/tts/stream",
        help="stream endpoint path (default: /tts/stream)",
    )
    parser.add_argument(
        "--text",
        default="春の風に、桜の花びら、舞い上がる。川面きらめき、遠く霞む山、静かな午後。誰もいない道、足音だけが響く、春のひととき。",
        help="text to synthesize",
    )
    parser.add_argument(
        "--language",
        default="JP",
        choices=["ZH", "JP", "EN", "MIX", "AUTO"],
        help="language id",
    )
    parser.add_argument("--speaker-name", default="", help="speaker name from config.json")
    parser.add_argument("--speaker", type=int, default=None, help="speaker id")
    parser.add_argument("--length-scale", type=float, default=None)
    parser.add_argument("--noise-scale", type=float, default=None)
    parser.add_argument("--max-chunk-chars", type=int, default=None)
    parser.add_argument("--chunk-pause-ms", type=int, default=None)
    parser.add_argument(
        "--save",
        default="",
        help="optional path to save received PCM (e.g. output/stream_live.pcm)",
    )
    parser.add_argument(
        "--read-size",
        type=int,
        default=8192,
        help="socket read size in bytes (default: 8192)",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="receive stream only, do not play audio",
    )
    return parser.parse_args()


def build_payload(args: argparse.Namespace) -> dict:
    payload: dict = {
        "text": args.text,
        "language": args.language,
    }
    if args.speaker_name:
        payload["speaker_name"] = args.speaker_name
    if args.speaker is not None:
        payload["speaker"] = args.speaker
    if args.length_scale is not None:
        payload["length_scale"] = args.length_scale
    if args.noise_scale is not None:
        payload["noise_scale"] = args.noise_scale
    if args.max_chunk_chars is not None:
        payload["max_chunk_chars"] = args.max_chunk_chars
    if args.chunk_pause_ms is not None:
        payload["chunk_pause_ms"] = args.chunk_pause_ms
    return payload


def default_server_cmd() -> str:
    if sys.platform == "win32":
        return ".\\bin\\windows\\bert-vits2-project.exe --server --host 127.0.0.1 --port 7860"
    if sys.platform == "darwin":
        return "./bin/macos/bert-vits2-project --server --host 127.0.0.1 --port 7860"
    return "./bin/linux/bert-vits2-project --server --host 127.0.0.1 --port 7860"


def check_health(host: str, port: int) -> None:
    url = f"http://{host}:{port}/health"
    with urllib.request.urlopen(url, timeout=5) as resp:
        if resp.status != 200:
            raise RuntimeError(f"health check failed: HTTP {resp.status}")


def stream_and_play(args: argparse.Namespace) -> int:
    if sd is None and not args.no_play:
        print("sounddevice is required for playback: pip install sounddevice")
        return 1

    payload = build_payload(args)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Connection": "close",
    }

    print(f"POST http://{args.host}:{args.port}{args.path}")
    print(f"text={payload['text'][:64]}{'...' if len(payload['text']) > 64 else ''}")
    print(f"language={payload['language']}")

    conn = http.client.HTTPConnection(args.host, args.port, timeout=300)
    try:
        conn.request("POST", args.path, body, headers)
        resp = conn.getresponse()
        if resp.status != 200:
            err = resp.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {resp.status}: {err}")

        content_type = resp.getheader("Content-Type", "")
        if "l16" not in content_type.lower():
            raise RuntimeError(f"unexpected Content-Type: {content_type}")

        sample_rate = int(resp.getheader("X-Sample-Rate", "44100"))
        audio_format = resp.getheader("X-Audio-Format", "pcm_s16le")
        print(f"sample_rate={sample_rate} format={audio_format}")

        out_file = open(args.save, "wb") if args.save else None
        audio_out = None
        if not args.no_play:
            audio_out = sd.RawOutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
                blocksize=0,
            )
            audio_out.start()

        total_bytes = 0
        first_chunk_at: float | None = None
        started = time.monotonic()

        while True:
            chunk = resp.read(args.read_size)
            if not chunk:
                break
            if first_chunk_at is None:
                first_chunk_at = time.monotonic()
                ttfb_ms = (first_chunk_at - started) * 1000
                print(f"first audio chunk: {len(chunk)} bytes, ttfb={ttfb_ms:.0f} ms")

            total_bytes += len(chunk)
            if out_file is not None:
                out_file.write(chunk)
            if audio_out is not None:
                audio_out.write(chunk)

        elapsed = time.monotonic() - started
        duration_sec = total_bytes / 2 / sample_rate if sample_rate > 0 else 0.0
        print(
            f"done: bytes={total_bytes} audio={duration_sec:.2f}s "
            f"wall={elapsed:.2f}s chunks_received"
        )

        if audio_out is not None:
            audio_out.stop()
            audio_out.close()
        if out_file is not None:
            out_file.close()
            print(f"saved PCM: {args.save}")

        if total_bytes < 2:
            raise RuntimeError("empty audio stream")
        return 0
    finally:
        conn.close()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    args = parse_args()
    try:
        check_health(args.host, args.port)
    except urllib.error.URLError as exc:
        print(f"server not reachable: http://{args.host}:{args.port} ({exc})")
        print(f"start server: {default_server_cmd()}")
        return 1

    try:
        return stream_and_play(args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    except Exception as exc:
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
