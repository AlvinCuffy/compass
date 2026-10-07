#!/usr/bin/env python3
"""
Recover an audio file from a .txt upload.

Handles two cases:
  1. the bytes are literally the audio (you renamed song.mp3 -> song.txt and the
     uploader only looked at the extension). Detected by magic number.
  2. the file really is base64 text (see below).

The chat uploader accepts text/plain but not audio, so one way to get a track
across is to base64 it into a .txt and upload that. This decodes it back, and
works with the various shapes that text arrives in:

  * plain base64, with or without newlines and spaces
  * PEM-style output from Windows' built-in
        certutil -encode "Even Here.mp3" audio.txt
    (-----BEGIN CERTIFICATE----- headers and 64-column wrapping)
  * a `data:audio/mpeg;base64,...` URI
  * base64 embedded in a JSON string

The format is sniffed from the leading bytes, so the extension does not matter.

Usage
-----
  python3 from_base64.py audio.txt                 # writes even-here.mp3 etc.
  python3 from_base64.py audio.txt --ext mp3 -o music.mp3
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import re
import subprocess
import sys

MAGIC = [
    (b"ID3", "mp3"), (b"\xff\xfb", "mp3"), (b"\xff\xf3", "mp3"), (b"\xff\xf2", "mp3"),
    (b"RIFF", "wav"), (b"fLaC", "flac"), (b"OggS", "ogg"), (b"FORM", "aiff"),
    (b"\x1a\x45\xdf\xa3", "webm"),
]


def sniff(data: bytes) -> str:
    for magic, ext in MAGIC:
        if data.startswith(magic):
            if ext == "wav" and data[8:12] == b"WAVE":
                return "wav"
            if ext != "wav":
                return ext
    if data[4:8] == b"ftyp":
        brand = data[8:12]
        return {"M4A ": "m4a", "mp42": "m4a", "isom": "m4a", "qt  ": "mov"}.get(
            brand.decode("latin-1"), "m4a")
    if data[:2] in (b"\xff\xf1", b"\xff\xf9"):
        return "aac"
    return "bin"


def strip_to_base64(text: str) -> str:
    """Pull a base64 payload out of whatever wrapper the text arrived in."""
    text = text.strip()
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            for key in ("content", "data", "base64", "audio"):
                if isinstance(obj.get(key), str):
                    text = obj[key]
                    break
        except json.JSONDecodeError:
            pass
    m = re.search(r"data:[\w/+.-]+;base64,([A-Za-z0-9+/=\s]+)", text)
    if m:
        text = m.group(1)
    text = re.sub(r"-----BEGIN [^-]+-----", "", text)
    text = re.sub(r"-----END [^-]+-----", "", text)
    text = re.sub(r"[\s\r\n]+", "", text)
    return text


def looks_binary(data: bytes) -> bool:
    """Raw audio masquerading as text: NUL bytes, or a known audio magic number."""
    if any(data.startswith(m) for m, _ in MAGIC):
        return True
    if data[:4] == b"RIFF" or data[4:8] == b"ftyp":
        return True
    sample = data[:4096]
    if b"\x00" in sample:
        return True
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Decode base64 text back into an audio file")
    p.add_argument("src", help="the .txt/.md/.json file containing base64")
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--ext", default=None, help="force an extension instead of sniffing")
    args = p.parse_args(argv)

    blob = open(args.src, "rb").read()

    # case 1: the file *is* the audio, just badly named
    if looks_binary(blob):
        ext = args.ext or sniff(blob)
        out = args.out or (os.path.splitext(os.path.basename(args.src))[0] + "." + ext)
        with open(out, "wb") as f:
            f.write(blob)
        print(f"this file is not base64 - it is raw audio. Renamed a .{ext} "
              f"to .txt? Recovering it as-is.")
        print(f"wrote {out} ({len(blob)/1e6:.2f} MB)")
        _verify(out)
        return 0

    raw = blob.decode("utf-8", errors="replace")
    payload = strip_to_base64(raw)
    if not payload:
        raise SystemExit("no base64 found in that file")

    # tolerate a missing pad, which some editors/clipboards eat
    payload += "=" * (-len(payload) % 4)
    try:
        data = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError) as e:
        raise SystemExit(f"that does not decode as base64: {e}")
    if len(data) < 512:
        raise SystemExit(f"decoded only {len(data)} bytes - the paste was probably truncated")

    ext = args.ext or sniff(data)
    out = args.out or os.path.splitext(os.path.basename(args.src))[0] + "." + ext
    with open(out, "wb") as f:
        f.write(data)

    print(f"decoded {len(payload)} base64 chars -> {len(data)/1e6:.2f} MB as {ext}")
    print(f"wrote {out}")
    _verify(out)
    return 0


def _verify(out: str) -> None:
    # confirm it is real, playable audio
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        ff = get_ffmpeg_exe()
        res = subprocess.run([ff, "-v", "error", "-i", out, "-t", "5", "-f", "f32le",
                              "-ac", "1", "-ar", "16000", "-"], capture_output=True)
        if res.returncode == 0 and res.stdout:
            print(f"audio OK: first {len(res.stdout)//4/16000:.1f}s decoded cleanly")
            print(f"\nnext: python3 lipsync.py --image face.jpg --audio '{out}' --out talking.mp4")
        else:
            print("warning: ffmpeg could not decode that; the audio may be incomplete",
                  file=sys.stderr)
    except ImportError:
        pass


if __name__ == "__main__":
    sys.exit(main())
