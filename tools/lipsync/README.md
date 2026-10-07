# Photo → talking / singing video

Give it **one portrait photo** and **one audio file**, get back an MP4 where the
person's mouth moves with the music, their eyes blink, and their head bobs on
the beat.

```bash
./setup.sh                       # once: installs numpy, mediapipe, opencv, ffmpeg…
python3 make_demo_audio.py --out demo/demo_song.wav --seconds 24   # optional test track

python3 lipsync.py \
  --image portrait.jpg \
  --audio song.mp3 \
  --out talking.mp4
```

A 24-second clip takes about 20 seconds to render on this CPU (roughly 1.2×
real time), so pull just the section you want:

```bash
python3 lipsync.py --image portrait.jpg --audio song.mp3 --out chorus.mp4 \
  --start 41 --duration 20          # start 41s in, render 20s
```

## What it actually does

For every frame, the audio's **vocal-band energy** (250 Hz–4 kHz) and its
**spectral centroid** are turned into a landmark-accurate warp of the mouth:

| piece | how it works |
|---|---|
| jaw drop | the lower lip, chin and throat move down together, with the photo's own texture carried along |
| mouth opening | the dark aperture the drop reveals is painted in, with a hint of upper teeth tinted to match the photo's own sclera |
| vowel shape | brighter, more open vowels ("ah") get a wider aperture than dark ones ("oh", "oo") |
| corners | squeeze inward slightly as the mouth opens, closing cleanly at the commissures |
| blinks | every 3.4–6.2 s, or pass `--no-blink` |
| head bob | onsets are detected from spectral flux, so the head nods on the beat, with a slow sway underneath |
| camera | slow push-in, plus a subtle brightness pulse on the beat |

`--mode music` keeps the mouth shut and leaves everything else running, which is
handy for an instrumental section.

## Getting the audio in

The chat uploader accepts **only** `png/webp/jpeg/gif`, `txt/md/csv/html/xml/css/js`,
`json/xml`, and `pdf`. No audio, no video, no archives -- so a track cannot be
attached directly. Two scripts in here get one across:

**From a public GitHub repo** (works for any size, no conversion). Upload the file
to a public repo, then:

```bash
python3 fetch_audio.py --repo AlvinCuffy/audio-temp        # auto-picks the file
python3 fetch_audio.py --repo X --path "Even Here.mp3" -o music.mp3
python3 fetch_audio.py --user AlvinCuffy                   # scan all public repos
```

It pulls the bytes straight from the git object store and checks the SHA-1 against
GitHub's own recorded hash, so a truncated download fails loudly instead of
rendering something broken. Add `--render --image face.png --out-video out.mp4`
to go straight from fetch to finished video.

**From a `.txt`** (`from_base64.py`) handles both of these:

* you renamed `song.mp3` to `song.txt` and the uploader only checked the
  extension -- the bytes are the audio, and the magic number gives it away;
* the text really is base64 -- plain, or PEM-wrapped by Windows'
  `certutil -encode "Even Here.mp3" audio.txt`, or a `data:` URI, or a JSON string.

```bash
python3 from_base64.py audio.txt                 # extension is sniffed
python3 from_base64.py audio.txt --ext mp3 -o music.mp3
```

Base64 is ~33% larger than the file it carries, so for anything long convert to a
low-bitrate mono MP3 first -- the renderer only needs the vocal band, and 64 kbps
mono drives the mouth identically while being a fraction of the size.

## Options

| flag | default | meaning |
|---|---|---|
| `--mode` | `sing` | `sing` (mouth driven by vocals), `music` (no mouth, just motion) |
| `--strength` | `1.0` | mouth motion scale, clamped to 0.2–1.3 |
| `--drop` | `0.20` | widest jaw drop, as a fraction of mouth width |
| `--gate` | `0.10` | loudness below which the mouth stays shut |
| `--fps` | `25` | output frame rate |
| `--max-side` | `1280` | longest output edge |
| `--crf` | `18` | x264 quality (lower = better/bigger) |
| `--start`, `--duration` | – | render just one section of the track |
| `--no-blink`, `--no-teeth` | – | switch those off |
| `--dry-run` | – | report drive levels and face geometry without rendering |

## What makes a good input

* **Face the camera.** Front-on or up to ~20–30° turned works; a full profile
  does not.
* **Big enough.** Ideally the mouth is 60+ px wide in the photo. The script
  warns below ~26 px, where the motion turns to mush.
* **Mouth closed, neutral.** A closed lip seal is the starting point the motion
  warps from, and it opens the cleanest.
* **A vertical phone photo is fine** — EXIF orientation is honoured.

## Honest limitations

* **The mouth is envelope-driven, not phoneme-driven.** It tracks loudness and
  vowels, so it looks convincingly like singing or clear speech, but it is not
  articulating specific words. For word-level accuracy you need a trained model
  (Wav2Lip, SadTalker, LivePortrait, MuseTalk).
* Those models need weights hosted on Hugging Face / Civitai, which this
  sandbox's network cannot reach (only GitHub, npm and PyPI are open), and they
  want a GPU — this box has 2 CPU cores and no torch. Running them locally is
  the better route if you need true lip-sync.
* **One face per photo.** With several, the largest is used.
* **Use photos you have the right to use.** Animating a real person's face
  without their permission is a problem everywhere, and some platforms ban
  AI-generated likenesses of public figures outright.
