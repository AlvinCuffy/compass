#!/usr/bin/env python3
"""Synthesise a short sing-song test track (arpeggio + bass + a soft kick).

Only used to smoke-test the renderer when no real music is at hand.
  python3 make_demo_audio.py --out demo.wav --seconds 20
"""
import argparse
import numpy as np
import soundfile as sf

SR = 44100


def voice(f0: float, dur: float, harmonics=(1.0, 0.5, 0.28, 0.14, 0.08),
          gain: float = 0.22, vibrato: float = 0.006, detune: float = 0.0) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    vib = 1.0 + vibrato * np.sin(2 * np.pi * 5.1 * t)
    phase = 2 * np.pi * (f0 + detune) * np.cumsum(vib) / SR
    sig = np.zeros(n)
    for k, amp in enumerate(harmonics, start=1):
        sig += amp * np.sin(k * phase)
    attack = np.minimum(1.0, np.arange(n) / (0.030 * SR))
    release = np.minimum(1.0, (n - np.arange(n)) / (0.090 * SR))
    return sig * attack * release * gain


def pluck(f0: float, dur: float, gain: float = 0.5) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    sig = np.sin(2 * np.pi * f0 * t) * np.exp(-t * 9)
    sig += 0.4 * np.sin(2 * np.pi * 2 * f0 * t) * np.exp(-t * 16)
    return sig * gain


def kick(dur: float = 0.20, f_hi: float = 110.0, f_lo: float = 42.0) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = np.linspace(f_hi, f_lo, n)
    return 0.5 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 20)


def hat(dur: float = 0.05, gain: float = 0.12) -> np.ndarray:
    n = int(dur * SR)
    rng = np.random.default_rng(1)
    return (rng.standard_normal(n) * np.exp(-np.arange(n) / (0.008 * SR)) * gain).astype(np.float32)


N = lambda name: {'C': 0, 'Dm': 1, 'F': 2, 'G': 3}[name]
PROG = [('C', [261.63, 329.63, 392.00]), ('G', [196.00, 246.94, 293.66]),
        ('Dm', [220.00, 261.63, 329.63]), ('F', [174.61, 220.00, 261.63])]


def build(seconds: float, bpm: float = 108.0) -> np.ndarray:
    total = int(seconds * SR)
    mix = np.zeros(total + SR, dtype=np.float32)
    bar = 4 * 60.0 / bpm                     # 4 beats per bar
    eighth = bar / 8

    k, h = kick(), hat()
    bar_i = 0
    t = 0.0
    while t < seconds:
        root, chord = PROG[bar_i % len(PROG)]
        # bass on beats 1 and 3
        for b in (0.0, 2 * 60.0 / bpm):
            v = voice(chord[0] / 2, bar * 0.42, harmonics=(1.0, 0.35, 0.12), gain=0.30, vibrato=0.0)
            i = int((t + b) * SR)
            mix[i:i + len(v)] += v
        # arpeggio across the bar
        for e in range(8):
            note = chord[e % 3] * (2 if e >= 6 else 1)
            p = pluck(note, eighth * 1.6, 0.34)
            i = int((t + e * eighth) * SR)
            mix[i:i + len(p)] += p
        # kick on beats, hats offbeat
        for b in range(4):
            i = int((t + b * 60.0 / bpm) * SR)
            mix[i:i + len(k)] += k
            j = int((t + (b + 0.5) * 60.0 / bpm) * SR)
            mix[j:j + len(h)] += h
        t += bar
        bar_i += 1

    mix = mix[:total]
    mix /= (np.abs(mix).max() + 1e-9)
    return (mix * 0.92).astype(np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo.wav")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--bpm", type=float, default=108.0)
    a = ap.parse_args()
    sig = build(a.seconds, a.bpm)
    sf.write(a.out, sig, SR)
    print(f"wrote {a.out}: {len(sig)/SR:.2f}s, peak {np.abs(sig).max():.3f}")
