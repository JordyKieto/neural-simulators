"""Crop the simulator canvas out of a capture video and write a README GIF (needs ffmpeg).

    python tools/make_gif.py results/eval/captures/driving_gentle_s2 assets/driving.gif [--width 384] [--fps 15]

The capture directory comes from eval/browser_bench/capture.mjs. The scripted segment is the last
`record_seconds` of the page video, since recording stops right after the script ends.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--start", type=float, default=0.0, help="seconds into the scripted segment")
    parser.add_argument("--seconds", type=float, help="length (default: whole segment)")
    args = parser.parse_args()

    meta = json.loads((args.capture / "canvas.json").read_text())
    video = next(args.capture.glob("*.webm"))
    duration = float(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, check=True,
        ).stdout
    )
    start = duration - meta["record_seconds"] + args.start
    length = args.seconds or meta["record_seconds"] - args.start
    b = meta["box"]
    crop = f"crop={int(b['width'])}:{int(b['height'])}:{int(b['x'])}:{int(b['y'])}"
    vf = f"{crop},fps={args.fps},scale={args.width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{length:.2f}", "-i", str(video),
         "-vf", vf, "-loop", "0", str(args.out)],
        check=True,
    )
    print(f"{args.out}: {args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
