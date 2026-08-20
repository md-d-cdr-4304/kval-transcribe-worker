"""Containerized transcription worker.

Runs headless inside an Azure Container Apps Job, entirely within KI's
private network. Reads audio/video files from INPUT_DIR, transcribes each
via direct-upload to an Azure OpenAI transcription deployment (push, not a
storage URL-fetch -- see project history for why that distinction matters),
and writes plain-text transcripts to OUTPUT_DIR. No source file is ever
modified; only ephemeral chunks generated in a scratch directory are sent
to the API.
"""

import ipaddress
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_KEY = os.environ["AZURE_OPENAI_KEY"]
DEPLOYMENT = os.environ.get("AZURE_TRANSCRIBE_DEPLOYMENT", "kval-transcribe-diarize")
API_VERSION = os.environ.get("AZURE_OPENAI_TRANSCRIBE_API_VERSION", "2024-06-01")
LANGUAGE = os.environ.get("AZURE_TRANSCRIBE_LANGUAGE", "sv")

INPUT_DIR = os.environ.get("INPUT_DIR", "/mnt/data/input")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/mnt/data/output")
SCRATCH_DIR = os.environ.get("SCRATCH_DIR", "/tmp/kval-chunks")

CHUNK_SECONDS = int(os.environ.get("CHUNK_SECONDS", "480"))
AUDIO_EXTENSIONS = (".mp4", ".mp3", ".wav", ".m4a", ".mov")

BOUNDARY = "----kvalcontainerboundary"


def _fail(msg: str) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def _verify_private_endpoint() -> None:
    host = urlparse(AZURE_OPENAI_ENDPOINT).hostname or ""
    if not host:
        _fail(f"Invalid AZURE_OPENAI_ENDPOINT: {AZURE_OPENAI_ENDPOINT!r}")
    try:
        ips = sorted({info[4][0] for info in socket.getaddrinfo(host, None)})
    except OSError as exc:
        _fail(f"DNS resolution failed for {host}: {exc}")
        return
    non_private = [ip for ip in ips if not ipaddress.ip_address(ip).is_private]
    if non_private:
        _fail(
            f"{host} resolved to non-private IP(s) {non_private} -- refusing to send "
            "audio outside the private network boundary. Check VNet integration / "
            "private DNS zone linkage for this Container Apps environment."
        )
    print(f"[ok] {host} -> {ips} (private)")


def _chunk_audio(src_path: str, out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    for old in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, old))
    pattern = os.path.join(out_dir, "chunk_%04d.wav")
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", src_path,
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            "-f", "segment", "-segment_time", str(CHUNK_SECONDS), "-reset_timestamps", "1",
            pattern,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {src_path}: {result.stderr.strip()}")
    chunks = sorted(
        os.path.join(out_dir, n) for n in os.listdir(out_dir) if n.startswith("chunk_")
    )
    if not chunks:
        raise RuntimeError(f"ffmpeg produced no chunks for {src_path}")
    return chunks


def _build_multipart(fields: dict, file_bytes: bytes, filename: str) -> bytes:
    parts = []
    for name, value in fields.items():
        parts.append(
            f"--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
        )
    parts.append(
        f"--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
        f"Content-Type: audio/wav\r\n\r\n".encode()
        + file_bytes + b"\r\n"
    )
    parts.append(f"--{BOUNDARY}--\r\n".encode())
    return b"".join(parts)


def _transcribe_chunk(path: str, attempt: int = 1) -> str:
    url = (
        f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/{DEPLOYMENT}"
        f"/audio/transcriptions?api-version={API_VERSION}"
    )
    with open(path, "rb") as f:
        audio_bytes = f.read()
    body = _build_multipart(
        {"language": LANGUAGE, "response_format": "json"}, audio_bytes, os.path.basename(path)
    )
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("api-key", AZURE_OPENAI_KEY)
    req.add_header("Content-Type", f"multipart/form-data; boundary={BOUNDARY}")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode("utf-8")).get("text", "")
    except urllib.error.HTTPError as exc:
        if exc.code == 429 and attempt <= 5:
            wait = 20 * attempt
            print(f"  rate-limited, waiting {wait}s (attempt {attempt})...")
            time.sleep(wait)
            return _transcribe_chunk(path, attempt + 1)
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}")


def _transcribe_file(src_path: str) -> str:
    stem = os.path.splitext(os.path.basename(src_path))[0]
    chunk_dir = os.path.join(SCRATCH_DIR, stem)
    chunks = _chunk_audio(src_path, chunk_dir)
    print(f"  {len(chunks)} chunk(s)")
    parts = []
    for i, chunk_path in enumerate(chunks, 1):
        t0 = time.time()
        text = _transcribe_chunk(chunk_path)
        print(f"  [{i}/{len(chunks)}] {time.time() - t0:.1f}s -> {len(text)} chars")
        parts.append(text.strip())
    for c in chunks:
        os.remove(c)
    return "\n\n".join(parts)


def main() -> None:
    _verify_private_endpoint()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(INPUT_DIR):
        _fail(f"INPUT_DIR does not exist: {INPUT_DIR}")

    pending = sorted(
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(AUDIO_EXTENSIONS)
    )
    if not pending:
        print(f"No audio files found in {INPUT_DIR}. Nothing to do.")
        return

    print(f"Found {len(pending)} file(s) in {INPUT_DIR}")
    for name in pending:
        stem = os.path.splitext(name)[0]
        out_path = os.path.join(OUTPUT_DIR, f"{stem}.txt")
        if os.path.exists(out_path):
            print(f"Skipping {name} (already transcribed)")
            continue
        print(f"Transcribing: {name}")
        src_path = os.path.join(INPUT_DIR, name)
        try:
            text = _transcribe_file(src_path)
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            continue
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  saved -> {out_path} ({len(text)} chars)")

    print("Done.")


if __name__ == "__main__":
    main()
