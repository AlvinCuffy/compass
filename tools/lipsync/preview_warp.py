#!/usr/bin/env python3
"""
QA a portrait *before* committing to a render.

Renders the mouth warp at several intensities straight to a PNG contact sheet,
so you can see whether the geometry holds up -- whether the jaw smears, whether
a nearby object (a mic, a pop filter, someone's shoulder) gets dragged along,
whether the opening reads as a mouth -- without waiting for a video.

It also reports the numbers that predict trouble: how far the face is turned,
how wide the mouth is in pixels, and whether the warp reaches the edge of its
own region (which would show as a seam).

Usage
-----
  python3 preview_warp.py --image portrait.jpg
  python3 preview_warp.py --image portrait.jpg --out check.png --zoom 2.5
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

import lipsync as L


def turn_angle(pts: np.ndarray) -> float:
    """Rough yaw of the head in degrees, from how the face is foreshortened."""
    lc, rc = pts[L.L_MOUTH_CORNER], pts[L.R_MOUTH_CORNER]
    nose = pts[1]
    mid = (lc + rc) * 0.5
    width = float(np.linalg.norm(rc - lc))
    # horizontal offset of the nose from the mouth's midpoint, as a fraction of
    # mouth width: grows as the head turns away from the camera
    off = float(nose[0] - mid[0]) / max(width, 1e-3)
    return float(np.degrees(np.arcsin(np.clip(off * 1.6, -1.0, 1.0))))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Preview the mouth warp on a still")
    p.add_argument("--image", required=True)
    p.add_argument("--out", default="warp_preview.png")
    p.add_argument("--zoom", type=float, default=2.2)
    p.add_argument("--levels", default="0,0.25,0.55,0.85,1.0",
                   help="mouth-open intensities to show")
    p.add_argument("--max-side", type=int, default=1280)
    p.add_argument("--drop", type=float, default=0.20)
    args = p.parse_args(argv)

    img = L.load_image(args.image)
    h0, w0 = img.shape[:2]
    if max(h0, w0) > args.max_side:
        s = args.max_side / max(h0, w0)
        img = cv2.resize(img, (int(round(w0 * s)), int(round(h0 * s))),
                         interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]

    pts = L.detect_landmarks(img)
    if pts is None:
        print("FAIL: no face detected in this image", file=sys.stderr)
        return 1

    mouth = L.build_mouth_fields(pts, img.shape, max_drop_frac=args.drop)
    mouth["teeth_bgr"] = L.sample_teeth_colour(img, pts)
    mw = mouth["mouth_w"]
    yaw = turn_angle(pts)

    print(f"image        : {w}x{h}")
    print(f"mouth width  : {mw:.1f} px")
    print(f"max jaw drop : {mouth['max_drop']:.1f} px")
    print(f"head turn    : ~{abs(yaw):.0f}deg {'(near-frontal)' if abs(yaw)<20 else '(three-quarter)' if abs(yaw)<45 else '(profile - expect trouble)'}")

    # how much of the available jaw travel we actually use, and whether the
    # warp is still non-zero where its region ends (a seam risk)
    print(f"warp at region edge: {float(np.abs(np.concatenate([mouth['down'][0], mouth['down'][-1], mouth['down'][:, 0], mouth['down'][:, -1]])).max()):.4f} (want 0)")
    if mw < 26:
        print("WARN: mouth under ~26px - motion will be mushy. Use a closer photo.")

    levels = [float(v) for v in args.levels.split(",")]
    x0, ya, x1, yb = mouth["roi"]
    cy = int(mouth["y_lip"])
    cx = int(mouth["cx"])
    # a generous crop so nearby objects (mic, hands) are visible in the preview
    cw = int(max(mw * 3.4, 220))
    ch = int(cw * 0.72)
    tx0, ty0 = max(0, cx - cw // 2), max(0, cy - ch // 2)
    tx1, ty1 = min(w, tx0 + cw), min(h, ty0 + ch)

    tiles = []
    for a in levels:
        frame = img.copy()
        if a > 1e-4:
            mx = mouth["base_x"] + a * mouth["corner_in"] * mouth["inward"]
            my = mouth["base_y"] - a * mouth["max_drop"] * mouth["down"]
            frame[ya:yb, x0:x1] = cv2.remap(
                frame, mx.astype(np.float32), my.astype(np.float32),
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            L.paint_mouth(frame, mouth, a, True, vowel=0.5)
        crop = frame[ty0:ty1, tx0:tx1]
        crop = cv2.resize(crop, None, fx=args.zoom, fy=args.zoom,
                          interpolation=cv2.INTER_CUBIC)
        label = "original" if a == 0 else f"a={a:.2f}"
        cv2.putText(crop, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 200, 0) if a == 0 else (0, 0, 255), 2)
        tiles.append(crop)

    hmin = min(t.shape[0] for t in tiles)
    per_row = 3
    rows = []
    for i in range(0, len(tiles), per_row):
        chunk = [t[:hmin] for t in tiles[i:i + per_row]]
        while len(chunk) < per_row and len(tiles) > 1:
            chunk.append(np.zeros_like(chunk[0]))
        rows.append(np.hstack(chunk))
    cv2.imwrite(args.out, np.vstack(rows))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
