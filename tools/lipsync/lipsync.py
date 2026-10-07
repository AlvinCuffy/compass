#!/usr/bin/env python3
"""
Turn a single portrait photo + an audio track into a "talking / singing" video.

How it works
------------
For every video frame the audio's vocal-band energy (plus spectral flatness,
which separates vowel-like sound from noise-like sound) drives a
landmark-accurate warp of the photo's mouth region:

  * the upper lip stays planted, the lower lip and chin drop with a smooth
    depth falloff (so the neck and collar never smear),
  * the dark mouth aperture that the drop reveals is painted in, with a hint of
    teeth, so a closed-mouth photo actually reads as an open mouth,
  * the mouth corners pull inward a little as it opens,
  * eyes blink on a natural schedule,
  * head bobs on detected onsets (beats) with a slow sway,
  * slow camera push-in and a light brightness pulse on the beat.

Honest description: this is a *puppet-style* audio-driven animation. The mouth
motion is envelope-driven, not phoneme-driven, so it tracks loudness and
vowels rather than specific words. It looks convincing on singing and clear
speech, especially at typical social-video sizes. It is not a generative
lip-sync model (Wav2Lip / SadTalker / LivePortrait class) -- those need model
weights this sandbox's network cannot reach.

Usage
-----
  python3 lipsync.py --image face.jpg --audio song.mp3 --out out.mp4
  python3 lipsync.py --image face.jpg --audio song.mp3 --start 34 --duration 20
  python3 lipsync.py --image face.jpg --audio song.mp3 --mode music    # no mouth
  python3 lipsync.py --image face.jpg --audio song.mp3 --dry-run

Modes
-----
  sing   mouth driven by the vocal band (default; use for vocals / speech)
  music  mouth stays shut; head bob, sway, camera and light still react
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile

import cv2
import mediapipe as mp
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe
from scipy.signal import butter, sosfilt


# --------------------------------------------------------------------------
# mediapipe FaceMesh landmark indices
# --------------------------------------------------------------------------
L_MOUTH_CORNER, R_MOUTH_CORNER = 61, 291
UPPER_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 308, 415]
LOWER_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 415]
CHIN = 152
NOSE_BASE = 2

L_EYE_UPPER = [159, 158, 157, 160, 161, 246]
L_EYE_LOWER = [145, 144, 153, 154, 155, 163]
R_EYE_UPPER = [386, 385, 384, 387, 388, 466]
R_EYE_LOWER = [374, 373, 380, 381, 382, 390]

APERTURE_BGR = np.array([16.0, 12.0, 30.0], dtype=np.float32)   # dark maroon
TEETH_BGR = np.array([205.0, 214.0, 222.0], dtype=np.float32)   # warm white


def smoothstep(a, b, x) -> np.ndarray:
    """0 below a, 1 above b, C1-smooth in between (a may be > b to flip).

    `a` and `b` may be scalars or arrays broadcast against `x`.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    span = b - a
    flat = np.abs(span) < 1e-9
    t = np.clip((x - a) / np.where(flat, 1.0, span), 0.0, 1.0)
    t = np.where(flat, (x >= b).astype(np.float32), t)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def exp_relax(target: np.ndarray, attack: float, release: float, fps: float) -> np.ndarray:
    """Asymmetric one-pole smoothing: fast attack, slower release (natural lips)."""
    dt = 1.0 / fps
    ka = 1.0 - math.exp(-dt / max(attack, 1e-4))
    kr = 1.0 - math.exp(-dt / max(release, 1e-4))
    out = np.empty_like(target)
    cur = 0.0
    for i in range(len(target)):
        t = float(target[i])
        cur += (t - cur) * (ka if t > cur else kr)
        out[i] = cur
    return out


# --------------------------------------------------------------------------
# audio -> per-frame drive signals
# --------------------------------------------------------------------------
def decode_audio(path: str, sr: int = 16000) -> np.ndarray:
    ffmpeg = get_ffmpeg_exe()
    cmd = [ffmpeg, "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def stft_mag(x: np.ndarray, n_fft: int = 512, hop: int = 128) -> np.ndarray:
    win = np.hanning(n_fft).astype(np.float32)
    frames = 1 + max(0, (len(x) - n_fft) // hop)
    out = np.empty((frames, n_fft // 2 + 1), dtype=np.float32)
    for i in range(frames):
        seg = x[i * hop: i * hop + n_fft]
        out[i] = np.abs(np.fft.rfft(seg * win)).astype(np.float32)
    return out


def audio_features(x: np.ndarray, fps: float, sr: int = 16000, gate: float = 0.10) -> dict:
    """Per-video-frame vocal energy, spectral flatness and onset strength."""
    sos = butter(4, [250.0, 4000.0], btype="bandpass", fs=sr, output="sos")
    vocal = sosfilt(sos, x).astype(np.float32)

    duration = len(x) / sr
    n = max(1, int(math.floor(duration * fps)))
    lead, lag = 0.075, 0.015          # RMS window around each frame's moment
    rms_v = np.zeros(n, dtype=np.float32)
    flat = np.zeros(n, dtype=np.float32)
    for i in range(n):
        t = i / fps
        a = max(0, int((t - lead) * sr))
        b = max(a + 1, min(len(x), int((t + lag) * sr)))
        seg = vocal[a:b]
        rms_v[i] = float(np.sqrt(np.mean(seg * seg)) + 1e-9)
        w = x[a:b]
        if len(w) >= 64:
            spec = np.abs(np.fft.rfft(w * np.hanning(len(w)))) + 1e-10
            spec = spec[1:]
            flat[i] = float(np.exp(np.mean(np.log(spec))) / (np.mean(spec) + 1e-10))
        else:
            flat[i] = 0.5

    db = 20.0 * np.log10(rms_v)
    lo, hi = np.percentile(db, 12), np.percentile(db, 97.5)
    if hi - lo < 1e-3:
        hi = lo + 1.0
    open_raw = np.clip((db - lo) / (hi - lo), 0.0, 1.0)
    open_raw = np.clip((open_raw - gate) / max(1.0 - gate, 1e-3), 0.0, 1.0) ** 0.85
    # noise-like frames (fricatives) open the mouth less than vowels
    open_raw *= (1.0 - 0.45 * np.clip((flat - 0.03) / 0.30, 0.0, 1.0)).astype(np.float32)
    open_curve = exp_relax(open_raw, 0.045, 0.13, fps)

    mag = stft_mag(x)
    flux = np.zeros(len(mag), dtype=np.float32)
    flux[1:] = np.maximum(0.0, mag[1:] - mag[:-1]).sum(axis=1)
    pos = np.clip((np.arange(n) / fps) / (128 / sr), 0, len(flux) - 1).astype(int)
    onset = flux[pos] / (np.percentile(flux[pos], 99) + 1e-9)
    onset = exp_relax(np.clip(onset, 0.0, 1.5), 0.02, 0.16, fps)

    # spectral centroid: a rough vowel hint -- dark, low-centred notes (o/u) get
    # a narrower aperture than bright, open ones (a/e/i)
    freqs = np.linspace(0, sr / 2, mag.shape[1], dtype=np.float32)
    denom = mag.sum(axis=1) + 1e-9
    centroid = (mag * freqs).sum(axis=1) / denom
    c = centroid[pos]
    c_lo, c_hi = np.percentile(c, 10), np.percentile(c, 90)
    if c_hi - c_lo < 1e-3:
        c_hi = c_lo + 1.0
    vowel = exp_relax(np.clip((c - c_lo) / (c_hi - c_lo), 0.0, 1.0), 0.06, 0.10, fps)

    return {
        "open": open_curve.astype(np.float32),
        "onset": onset.astype(np.float32),
        "vowel": vowel.astype(np.float32),
        "n_frames": n,
        "vocal_level_db": (float(lo), float(hi)),
    }


# --------------------------------------------------------------------------
# face geometry -> warp fields (computed once per image)
# --------------------------------------------------------------------------
def load_image(path: str) -> np.ndarray:
    """Read an image the right way up.

    cv2.imread ignores EXIF orientation, so phone photos come in sideways and
    face detection simply fails. Prefer Pillow (EXIF-aware), fall back to
    OpenCV if it is unavailable.
    """
    try:
        from PIL import Image, ImageOps
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            rgb = np.asarray(im.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"Could not read image: {path}")
        return img


def detect_landmarks(img_bgr: np.ndarray):
    h, w = img_bgr.shape[:2]
    scale = min(1.0, 1280.0 / max(h, w))
    small = cv2.resize(img_bgr, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA) if scale < 1 else img_bgr
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=4, refine_landmarks=True,
        min_detection_confidence=0.4,
    ) as mesh:
        res = mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    best = max(res.multi_face_landmarks, key=_face_area)
    return np.array([[lm.x * small.shape[1] / scale, lm.y * small.shape[0] / scale]
                     for lm in best.landmark], dtype=np.float32)


def _face_area(fl) -> float:
    xs = [lm.x for lm in fl.landmark]
    ys = [lm.y for lm in fl.landmark]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def _grid(x0, y0, x1, y1):
    xs = np.arange(x0, x1, dtype=np.float32)[None, :]
    ys = np.arange(y0, y1, dtype=np.float32)[:, None]
    return (np.broadcast_to(xs, (y1 - y0, x1 - x0)).copy(),
            np.broadcast_to(ys, (y1 - y0, x1 - x0)).copy())


def build_mouth_fields(pts: np.ndarray, shape, max_drop_frac: float = 0.20) -> dict:
    """Warp kernels for a jaw drop, plus the lip seal used to paint the aperture.

    The lower face is modelled as two layers:

      * a hairline *step* at the closed-mouth lip seal, restricted to the mouth
        itself -- everything below the seal moves down with the jaw, the upper
        lip stays planted,
      * a broad *deep* drop that ramps in below the lip and fades out under the
        chin, so the throat and collar are never dragged.

    Together they translate the lower lip almost rigidly (no smearing), and the
    dark aperture revealed by the step is painted in by `paint_mouth`.
    """
    h, w = shape[:2]
    lc, rc = pts[L_MOUTH_CORNER], pts[R_MOUTH_CORNER]
    cx = float((lc[0] + rc[0]) * 0.5)
    mw = float(np.linalg.norm(rc - lc))                       # full mouth width
    y_lip = float(pts[[13, 14], 1].mean())                    # the seal
    y_chin = float(pts[CHIN, 1])
    y_nose = float(pts[NOSE_BASE, 1])

    x0 = int(max(0, math.floor(cx - 1.45 * mw)))
    x1 = int(min(w, math.ceil(cx + 1.45 * mw)))
    y0 = int(max(0, math.floor(y_nose - 0.10 * mw)))
    y1 = int(min(h, math.ceil(y_chin + 0.78 * mw)))
    roi = (x0, y0, x1, y1)
    gx, gy = _grid(x0, y0, x1, y1)
    dx = np.abs(gx - cx)

    # taper across the lips (0 at the commissures) vs. a wider taper for the jaw
    h_mouth = 1.0 - smoothstep(0.34 * mw, 0.48 * mw, dx)
    h_jaw = 1.0 - smoothstep(0.55 * mw, 1.15 * mw, dx)

    # the jaw motion has to be spent by the bottom of the throat, or the collar
    # gets dragged; keeping the fade inside the ROI is what avoids a seam there
    fade = (1.0 - smoothstep(y_chin - 0.25 * mw, y_chin + 0.75 * mw, gy)).astype(np.float32)

    # hairline step at the seal, smoothed over ~1.5px purely to kill aliasing
    v_step = smoothstep(y_lip - 0.008 * mw, y_lip + 0.008 * mw, gy)
    # deep jaw drop: in below the lip, out again under the chin
    v_deep = smoothstep(y_lip - 0.02 * mw, y_lip + 0.38 * mw, gy)

    # max, not sum: both terms describe the same jaw motion, the step just owns
    # the lip seal while the deep term owns the chin and its sides
    down = np.maximum(h_mouth * v_step, h_jaw * v_deep) * fade
    down = down.astype(np.float32)

    # corners squeeze inward a little as the jaw drops
    corner_band = (smoothstep(y_lip - 0.30 * mw, y_lip, gy)
                   * (1.0 - smoothstep(y_lip + 0.05 * mw, y_lip + 0.55 * mw, gy)))
    inward = (np.sign(cx - gx) * corner_band * h_mouth).astype(np.float32)

    return {
        "roi": roi, "down": down, "inward": inward,
        "base_x": gx, "base_y": gy,
        "mouth_w": mw, "cx": cx, "y_lip": y_lip, "y_chin": y_chin, "y_nose": y_nose,
        "max_drop": max_drop_frac * mw,
        "corner_in": 0.10 * mw,
    }


def sample_field(field: np.ndarray, roi, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Read the warp field at arbitrary image coordinates (0 outside the ROI)."""
    x0, y0, x1, y1 = roi
    ix = np.clip((x - x0).astype(np.int32), 0, field.shape[1] - 1)
    iy = np.clip((y - y0).astype(np.int32), 0, field.shape[0] - 1)
    inside = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    return np.where(inside, field[iy, ix], 0.0).astype(np.float32)


def build_eye_fields(pts: np.ndarray, shape) -> dict | None:
    """Compact-support kernel that slides the lid skin down over each eye."""
    h, w = shape[:2]
    fields = []
    for up_idx, lo_idx in ((L_EYE_UPPER, L_EYE_LOWER), (R_EYE_UPPER, R_EYE_LOWER)):
        up, lo = pts[up_idx], pts[lo_idx]
        ex = float((up[:, 0].min() + up[:, 0].max()) * 0.5)
        ew = float(up[:, 0].max() - up[:, 0].min())
        if ew < 2:
            continue
        y_up, y_lo = float(up[:, 1].mean()), float(lo[:, 1].mean())
        travel = float(max(y_lo - y_up, 0.05 * ew))
        x0 = int(max(0, math.floor(ex - 0.95 * ew)))
        x1 = int(min(w, math.ceil(ex + 0.95 * ew)))
        ya = int(max(0, math.floor(y_up - 1.7 * travel)))
        yb = int(min(h, math.ceil(y_lo + 0.40 * travel)))
        gx, gy = _grid(x0, ya, x1, yb)
        vert = smoothstep(y_up - 1.35 * travel, y_up - 0.10 * travel, gy)
        vert *= (1.0 - smoothstep(y_lo - 0.50 * travel, y_lo + 0.30 * travel, gy)).astype(np.float32)
        horiz = 1.0 - smoothstep(0.60 * ew, 1.0 * ew, np.abs(gx - ex))
        fields.append({"roi": (x0, ya, x1, yb), "weight": (vert * horiz).astype(np.float32),
                       "travel": travel, "base_x": gx, "base_y": gy})
    if not fields:
        return None
    return {
        "eyes": fields,
        "roi": (min(f["roi"][0] for f in fields), min(f["roi"][1] for f in fields),
                max(f["roi"][2] for f in fields), max(f["roi"][3] for f in fields)),
    }


def blink_curve(n: int, fps: float, seed: int = 7, every=(3.4, 6.2)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros(n, dtype=np.float32)
    t = float(rng.uniform(*every)) * 0.5
    while t < n / fps:
        i0 = int(t * fps)
        for k, amp in enumerate((0.35, 1.0, 0.55, 0.15)):
            j = i0 + k
            if 0 <= j < n:
                out[j] = max(out[j], amp)
        t += float(rng.uniform(*every))
    return exp_relax(out, 0.022, 0.055, fps)


def sample_teeth_colour(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Use the photo's own sclera as the tooth white, so the fill matches the
    lighting of the shot instead of glowing like a light bar."""
    h, w = img.shape[:2]
    samples = []
    for up_idx, lo_idx in ((L_EYE_UPPER, L_EYE_LOWER), (R_EYE_UPPER, R_EYE_LOWER)):
        up, lo = pts[up_idx], pts[lo_idx]
        x0 = int(max(0, up[:, 0].min() + 2)); x1 = int(min(w, up[:, 0].max() - 2))
        y0 = int(max(0, up[:, 1].mean())); y1 = int(min(h, lo[:, 1].mean() + 1))
        if x1 - x0 < 3 or y1 - y0 < 2:
            continue
        patch = img[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
        if len(patch):
            samples.append(np.percentile(patch, 88, axis=0))
    if not samples:
        return TEETH_BGR.copy()
    colour = np.mean(samples, axis=0)
    return np.clip(colour * 0.97 - 6.0, 90.0, 245.0).astype(np.float32)   # a touch under sclera


def _feather(ph: int, pw: int, pad: int = 4) -> np.ndarray:
    ax = np.minimum(np.arange(pw), pw - 1 - np.arange(pw)).astype(np.float32)[None, :]
    ay = np.minimum(np.arange(ph), ph - 1 - np.arange(ph)).astype(np.float32)[:, None]
    t = np.clip(np.minimum(ax, ay) / max(pad, 1), 0.0, 1.0)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def paint_mouth(frame: np.ndarray, mouth: dict, a: float, teeth: bool,
                vowel: float = 0.5) -> None:
    """Paint the dark aperture the jaw drop revealed, with a hint of upper teeth.

    The opening is described as a pair of curves across the mouth: a nearly flat
    upper edge pinned just below the (planted) upper lip, and an elliptical lower
    edge that meets it again at the commissures. Both edges are smooth, so the
    hole closes gradually at the corners instead of ending in a hard cut, and the
    bottom edge deliberately stops just short of where the lower lip landed --
    the skin left in between reads as the inner lower lip.

    Defaults are deliberately restrained (a small, dark, slightly irregular
    opening) because a restrained fake mouth reads as far more convincing than a
    large cartoon one.
    """
    if a <= 1e-4:
        return
    drop, mw, cx = mouth["max_drop"], mouth["mouth_w"], mouth["cx"]
    y_lip = mouth["y_lip"]
    h, w = frame.shape[:2]

    rx = 0.30 * mw * (0.84 + 0.32 * vowel)    # brighter vowels open wider
    half = 0.95 * a * drop                    # depth at the centre

    x0 = int(max(0, math.floor(cx - rx - 4)))
    x1 = int(min(w, math.ceil(cx + rx + 4)))
    y0 = int(max(0, math.floor(y_lip - 4)))
    y1 = int(min(h, math.ceil(y_lip + half + 4)))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return

    gx = np.arange(x0, x1, dtype=np.float32)[None, :]
    gy = np.arange(y0, y1, dtype=np.float32)[:, None]
    t = np.clip((gx - cx) / rx, -1.0, 1.0)
    t2 = t * t
    side = 1.0 - smoothstep(0.80, 1.0, np.abs(t))        # closes the ends softly

    # an organic wobble on the edges so the opening is never a perfect ellipse
    wob = 1.0 + 0.035 * np.sin(31.0 * t + 0.7) + 0.02 * np.sin(57.0 * t + 2.1)
    top = y_lip + 0.05 * a * drop * (1.0 - t2) * wob     # never above the lip line
    gap = a * drop * (0.92 - 0.88 * t2) * wob            # widest at the centre
    bot = top + gap
    if float(gap.max()) < 1.4:
        return

    mask = (smoothstep(top, top + 1.2, gy)
            * (1.0 - smoothstep(bot - 1.2, bot, gy))
            * side).astype(np.float32)
    if mask.max() <= 0:
        return

    region = frame[y0:y1, x0:x1].astype(np.float32)
    hole = region * 0.22
    depth = np.linspace(0.80, 1.15, y1 - y0, dtype=np.float32)[:, None, None]
    hole = hole + APERTURE_BGR * 0.62 * depth

    if teeth:
        # upper teeth hug the top edge, thinning and fading toward the sides
        te_top = top + 0.08 * gap
        te_bot = top + 0.50 * gap
        tmask = (smoothstep(te_top, te_top + 1.0, gy)
                 * (1.0 - smoothstep(te_bot - 1.6, te_bot, gy))
                 * (1.0 - smoothstep(0.10, 0.72, np.abs(t)))
                 * smoothstep(0.45, 0.95, mask)).astype(np.float32)
        tmask = cv2.GaussianBlur(tmask, (3, 3), 0.7)
        tmask *= smoothstep(2.5, 7.5, float(gap.max())) * min(1.0, a * 1.15) * 0.62
        lit = (mouth.get("teeth_bgr", TEETH_BGR) * 0.93)[None, None, :]
        hole = hole * (1.0 - tmask[:, :, None]) + lit * tmask[:, :, None]

    out = region * (1.0 - mask[:, :, None]) + hole * mask[:, :, None]
    frame[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# renderer
# --------------------------------------------------------------------------
def render(image_path: str, audio_path: str, out_path: str, *, mode: str = "sing",
           fps: float = 25.0, max_side: int = 1280, strength: float = 1.0,
           start: float = 0.0, duration: float | None = None, blinks: bool = True,
           teeth: bool = True, gate: float = 0.10, drop: float = 0.20,
           dry_run: bool = False, crf: int = 18) -> dict:
    img = load_image(image_path)
    h0, w0 = img.shape[:2]
    if max(h0, w0) > max_side:
        s = max_side / max(h0, w0)
        img = cv2.resize(img, (int(round(w0 * s)), int(round(h0 * s))), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]

    pts = detect_landmarks(img)
    if pts is None:
        raise SystemExit("No face found. I need a front-facing photo with a visible mouth.")

    mouth = build_mouth_fields(pts, img.shape, max_drop_frac=drop)
    mouth["teeth_bgr"] = sample_teeth_colour(img, pts)
    if mouth["mouth_w"] < 26:
        print(f"warning: mouth is only {mouth['mouth_w']:.0f}px wide - use a "
              f"larger/closer photo (at least ~{130 / mouth['mouth_w']:.1f}x this "
              f"face size) for a convincing result", file=sys.stderr)
    eyes = build_eye_fields(pts, img.shape) if blinks else None
    face_cx, face_cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())

    audio = decode_audio(audio_path)
    sr = 16000
    if start > 0:
        audio = audio[int(start * sr):]
    if duration:
        audio = audio[: int(duration * sr)]
    if len(audio) < sr // 4:
        raise SystemExit("Audio is empty after trimming.")

    feats = audio_features(audio, fps, sr, gate)
    n = feats["n_frames"]
    open_curve, onset = feats["open"], feats["onset"]
    if mode not in ("sing", "music", "auto"):
        raise SystemExit(f"Unknown mode: {mode}")

    if dry_run:
        return {"frames": n, "size": f"{w}x{h}", "mode": mode,
                "mouth_open_peak": round(float(open_curve.max()), 3),
                "mouth_open_mean": round(float(open_curve.mean()), 3),
                "mouth_width_px": round(mouth["mouth_w"], 1),
                "max_jaw_drop_px": round(mouth["max_drop"], 1)}

    # beyond ~1.3 the aperture would exceed the lip-to-chin gap and the mouth
    # starts to look like a hole punched in the face
    strength = max(0.2, min(float(strength), 1.3))
    play = open_curve * strength if mode in ("sing", "auto") else np.zeros(n, dtype=np.float32)
    vowel_c = feats["vowel"]
    blink_c = blink_curve(n, fps) if eyes else np.zeros(n, dtype=np.float32)

    ffmpeg = get_ffmpeg_exe()
    tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    enc = subprocess.Popen(
        [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", "-an",
         "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
         "-pix_fmt", "yuv420p", tmp_video],
        stdin=subprocess.PIPE)

    x0, ya, x1, yb = mouth["roi"]
    max_drop, corner_in = mouth["max_drop"], mouth["corner_in"]

    for i in range(n):
        frame = img.copy()
        a = float(play[i])

        # --- jaw drop + mouth aperture ---------------------------------------
        if a > 1e-4:
            # dest -> src: material below the seal moves *down*, so sample from
            # above it; the corners squeeze inward slightly.
            mx = mouth["base_x"] + a * corner_in * mouth["inward"]
            my = mouth["base_y"] - a * max_drop * mouth["down"]
            frame[ya:yb, x0:x1] = cv2.remap(
                frame, mx.astype(np.float32), my.astype(np.float32),
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            paint_mouth(frame, mouth, a, teeth, vowel=float(vowel_c[i]))

        # --- blinks ----------------------------------------------------------
        if eyes is not None and blink_c[i] > 1e-3:
            b = float(blink_c[i])
            for eye in eyes["eyes"]:
                kx0, kya, kx1, kyb = eye["roi"]
                mx = eye["base_x"].astype(np.float32)
                my = (eye["base_y"] - b * eye["travel"] * eye["weight"]).astype(np.float32)
                warped = cv2.remap(frame, mx, my, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
                alpha = _feather(*warped.shape[:2])[:, :, None]
                dst = frame[kya:kyb, kx0:kx1]
                frame[kya:kyb, kx0:kx1] = (warped * alpha + dst * (1.0 - alpha)).astype(np.uint8)

        # --- head motion, camera, light --------------------------------------
        t = i / fps
        pulse = float(onset[i])
        sway = math.sin(2 * math.pi * 0.21 * t)
        sway2 = math.sin(2 * math.pi * 0.17 * t + 1.1)
        zoom = 1.030 + 0.016 * (t / max(n / fps, 1e-3)) + 0.013 * pulse
        ang = 0.85 * sway2
        tx = 0.008 * w * sway
        ty = -0.011 * h * pulse + 0.004 * h * math.sin(2 * math.pi * 0.13 * t + 0.4)
        m = cv2.getRotationMatrix2D((face_cx + tx, face_cy + ty), ang, zoom)
        frame = cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        frame = np.clip(frame.astype(np.float32) * (1.0 + 0.035 * pulse), 0, 255).astype(np.uint8)

        enc.stdin.write(frame.tobytes())

    enc.stdin.close()
    if enc.wait() != 0:
        raise SystemExit("ffmpeg encoding failed")

    subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-i", tmp_video, "-ss", f"{start}", "-i", audio_path,
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-shortest", "-movflags", "+faststart", out_path], check=True)
    os.unlink(tmp_video)

    return {"frames": n, "size": f"{w}x{h}", "mode": mode, "seconds": round(n / fps, 2),
            "mouth_open_peak": round(float(open_curve.max()), 3),
            "mouth_open_mean": round(float(open_curve.mean()), 3),
            "max_jaw_drop_px": round(max_drop, 1), "out": out_path}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Still photo + audio -> talking/singing video")
    p.add_argument("--image", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--out", default="out.mp4")
    p.add_argument("--mode", choices=["sing", "music", "auto"], default="sing")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--max-side", type=int, default=1280)
    p.add_argument("--strength", type=float, default=1.0, help="mouth motion scale")
    p.add_argument("--drop", type=float, default=0.20, help="max jaw drop as a fraction of mouth width")
    p.add_argument("--gate", type=float, default=0.10, help="loudness below which the mouth stays shut")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--no-blink", action="store_true")
    p.add_argument("--no-teeth", action="store_true")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    info = render(args.image, args.audio, args.out, mode=args.mode, fps=args.fps,
                  max_side=args.max_side, strength=args.strength, start=args.start,
                  duration=args.duration, blinks=not args.no_blink, teeth=not args.no_teeth,
                  gate=args.gate, drop=args.drop, dry_run=args.dry_run, crf=args.crf)
    for k, v in info.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
