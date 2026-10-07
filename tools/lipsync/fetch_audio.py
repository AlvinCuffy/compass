#!/usr/bin/env python3
"""
Fetch an audio file out of a public GitHub repo, byte-for-byte.

Why this exists: the chat uploader only accepts images, text, PDF and JSON/XML
-- no audio, video or archives -- so a track has to reach the sandbox some other
way. GitHub is reachable from here; Dropbox, Drive and WeTransfer are not.

The file is pulled straight from the git object store and its SHA-1 is checked
against GitHub's own recorded hash, so a truncated or mangled download is caught
rather than silently rendered.

Usage
-----
  python3 fetch_audio.py --repo AlvinCuffy/audio-temp
  python3 fetch_audio.py --repo AlvinCuffy/audio-temp --path "Even Here.mp3"
  python3 fetch_audio.py --user AlvinCuffy              # scan every public repo
  python3 fetch_audio.py --repo X --out music.mp3 --render --image face.png \
      --out-video talking.mp4 --duration 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

AUDIO_EXT = (".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".oga", ".opus",
             ".aif", ".aiff", ".wma", ".webm", ".mp4", ".mov", ".mkv")


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def gh_api(path: str, *, raw: bool = False, accept: str | None = None) -> bytes:
    """Call the GitHub API, preferring the authenticated `gh` CLI."""
    if have("gh"):
        cmd = ["gh", "api", path]
        if accept:
            cmd += ["-H", f"Accept: {accept}"]
        if not raw:
            cmd += ["--jq", "."]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.decode().strip() or "gh api failed")
        return res.stdout
    import urllib.request
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        headers={"Accept": accept or "application/vnd.github+json",
                 "User-Agent": "fetch-audio"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def repos_for(user: str) -> list[str]:
    data = json.loads(gh_api(f"/users/{user}/repos?per_page=100"))
    return [r["full_name"] for r in data if not r.get("private")]


def default_branch(repo: str) -> str:
    return json.loads(gh_api(f"/repos/{repo}"))["default_branch"]


def tree_audio(repo: str, branch: str) -> list[dict]:
    data = json.loads(gh_api(f"/repos/{repo}/git/trees/{branch}?recursive=1"))
    out = []
    for node in data.get("tree", []):
        if node.get("type") == "blob" and node["path"].lower().endswith(AUDIO_EXT):
            out.append({"path": node["path"], "sha": node["sha"], "size": node.get("size", 0)})
    return sorted(out, key=lambda n: -n["size"])


def download(repo: str, node: dict, out_path: str) -> str:
    """Pull one blob as raw bytes and verify it against GitHub's hash."""
    data = gh_api(f"/repos/{repo}/git/blobs/{node['sha']}",
                  raw=True, accept="application/vnd.github.raw")
    computed = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    if computed != node["sha"]:
        raise SystemExit(f"checksum mismatch for {node['path']}: "
                         f"got {computed}, expected {node['sha']}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(data)
    return computed


def probe(path: str, seconds: float = 5.0) -> dict:
    """Confirm the file really is decodable audio, and report what it is."""
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError:
        return {"decodable": None, "note": "imageio-ffmpeg not installed"}
    ff = get_ffmpeg_exe()
    res = subprocess.run(
        [ff, "-v", "error", "-i", path, "-t", str(seconds), "-f", "f32le",
         "-ac", "1", "-ar", "16000", "-"], capture_output=True)
    if res.returncode != 0 or not res.stdout:
        return {"decodable": False, "error": res.stderr.decode().strip()[:200]}
    return {"decodable": True, "sampled_seconds": len(res.stdout) // 4 / 16000}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Fetch audio from a public GitHub repo")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", help="owner/name")
    src.add_argument("--user", help="scan all public repos of this user")
    p.add_argument("--branch", default=None, help="defaults to the repo's default branch")
    p.add_argument("--path", default=None, help="exact path within the repo")
    p.add_argument("-o", "--out", default=None, help="where to save it")
    p.add_argument("--render", action="store_true", help="render immediately after fetching")
    p.add_argument("--image", default=None)
    p.add_argument("--out-video", default="out.mp4", help="rendered video path")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=None)
    args = p.parse_args(argv)

    repos = repos_for(args.user) if args.user else [args.repo]
    if not repos:
        raise SystemExit(f"no public repos found for {args.user}")

    for repo in repos:
        branch = args.branch or default_branch(repo)
        try:
            found = tree_audio(repo, branch)
        except RuntimeError as e:
            print(f"  {repo}: {e}", file=sys.stderr)
            continue
        if not found:
            if args.user:
                print(f"  {repo}: no audio files")
            continue
        if args.path:
            found = [n for n in found if n["path"] == args.path or
                     os.path.basename(n["path"]) == os.path.basename(args.path)]
            if not found:
                continue
        node = found[0]
        print(f"  {repo} @ {branch}: {len(found)} audio file(s), taking the largest")
        out = args.out or os.path.basename(node["path"])
        sha = download(repo, node, out)
        info = probe(out)
        print(f"  saved    : {out}  ({node['size']/1e6:.2f} MB)")
        print(f"  verified : sha1 {sha[:12]} matches GitHub's recorded hash")
        print(f"  audio    : {json.dumps(info)}")
        if not info.get("decodable"):
            raise SystemExit("that file is not decodable audio")
        if args.render:
            if not args.image:
                raise SystemExit("--render needs --image")
            cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "lipsync.py"),
                   "--image", args.image, "--audio", out, "--out", args.out_video,
                   "--start", str(args.start)]
            if args.duration:
                cmd += ["--duration", str(args.duration)]
            print("  rendering:", " ".join(cmd))
            subprocess.run(cmd, check=True)
        else:
            print(f"\n  next: python3 lipsync.py --image face.jpg --audio '{out}' "
                  f"--out talking.mp4")
        return 0

    if args.path:
        raise SystemExit(f"no audio file matching {args.path!r} found")
    raise SystemExit("no audio files found in any public repo")


if __name__ == "__main__":
    sys.exit(main())
