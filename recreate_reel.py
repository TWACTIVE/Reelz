#!/usr/bin/env python3
"""
recreate_reel.py — Instagram Reel recreation CLI

Usage:
    python recreate_reel.py <instagram_url> --clips ./my_camera_roll [options]

Options:
    --clips PATH        Folder of your .mp4/.mov clips (required)
    --threshold FLOAT   Scene-cut threshold 0.0–1.0 (default: 0.3)
    --output PATH       Output video path (default: output/recreated_reel.mp4)
    --skip-download     Reuse last downloaded video in downloads/
    --force-reindex     Ignore clip_index.json cache, re-tag all clips
"""

import argparse
import json
import os
import sys
import hashlib
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import anthropic

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
OUTPUT_DIR = BASE_DIR / "output"
TMP_DIR = BASE_DIR / "tmp"
CLIP_INDEX_PATH = BASE_DIR / "clip_index.json"

DOWNLOADS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
TMP_DIR.mkdir(exist_ok=True)


# ── helpers ────────────────────────────────────────────────────────────────────

def check_deps():
    missing = []
    for tool in ("yt-dlp", "ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            missing.append(tool)
    if missing:
        print("ERROR: missing required tools:", ", ".join(missing))
        print()
        print("Install with:")
        print("  brew install yt-dlp ffmpeg          # macOS")
        print("  sudo apt install ffmpeg && pip install yt-dlp  # Linux")
        sys.exit(1)


def run(cmd, check=True, capture=False):
    """Run a shell command, returning CompletedProcess."""
    kwargs = dict(text=True)
    if capture:
        kwargs["capture_output"] = True
    result = subprocess.run(cmd, **kwargs)
    if check and result.returncode != 0:
        stderr = result.stderr if capture else ""
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n{stderr}")
    return result


def file_hash(path: Path) -> str:
    """Fast partial hash: first 256KB + file size. Avoids reading whole file."""
    h = hashlib.md5()
    size = path.stat().st_size
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(262144))
    return h.hexdigest()[:12]


def video_duration(path: Path) -> float:
    r = run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", str(path)],
        capture=True,
    )
    info = json.loads(r.stdout)
    return float(info["format"]["duration"])


def extract_frames(video_path: Path, timestamps: list[float], out_dir: Path) -> list[Path]:
    """Extract one frame per timestamp, return list of PNG paths."""
    frames = []
    for i, ts in enumerate(timestamps):
        out = out_dir / f"frame_{i:03d}.png"
        run(
            ["ffmpeg", "-y", "-ss", str(ts), "-i", str(video_path),
             "-frames:v", "1", "-q:v", "2", str(out)],
            capture=True,
        )
        if out.exists():
            frames.append(out)
    return frames


def encode_image(path: Path) -> str:
    import base64
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def describe_shot(client: anthropic.Anthropic, frames: list[Path], duration: float) -> dict:
    """Ask Claude to describe a shot given its frames."""
    content = []
    for frame in frames:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": encode_image(frame),
            },
        })
    content.append({
        "type": "text",
        "text": (
            f"These {len(frames)} frames are evenly sampled from a video shot that is "
            f"{duration:.2f} seconds long. Describe this shot as JSON with exactly these keys:\n"
            '{"shot_type": "...", "energy_level": "low|medium|high", "setting": "...", '
            '"subject": "...", "notable_action": "...", "duration_seconds": ' + str(round(duration, 2)) + "}\n"
            "shot_type examples: wide establishing, medium, close-up, aerial, POV, selfie, b-roll\n"
            "Respond with ONLY the JSON object, no markdown fences."
        ),
    })
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": content}],
    )
    text = msg.content[0].text.strip()
    # strip optional ```json fences in case the model adds them
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


# ── step 1: download ───────────────────────────────────────────────────────────

def download_reel(url: str, cookies_from_browser: str | None = None, cookies: str | None = None) -> Path:
    print(f"\n[1/8] Downloading reel from {url}")
    slug = hashlib.md5(url.encode()).hexdigest()[:8]
    out_template = str(DOWNLOADS_DIR / f"{slug}.%(ext)s")
    cmd = ["yt-dlp", "--no-playlist", "-f", "mp4/bestvideo+bestaudio/best",
           "--merge-output-format", "mp4", "-o", out_template]
    if cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]
    if cookies:
        cmd += [f"--cookies={cookies}"]
    cmd.append(url)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ERROR: yt-dlp failed.\n")
        print(r.stderr[-1500:])
        if "Private" in r.stderr or "login" in r.stderr.lower():
            print("Hint: this reel may be private or require login.")
        sys.exit(1)
    # find the file
    matches = list(DOWNLOADS_DIR.glob(f"{slug}.*"))
    mp4s = [m for m in matches if m.suffix in (".mp4", ".mkv", ".webm")]
    if not mp4s:
        print("ERROR: yt-dlp ran but no video file was found in", DOWNLOADS_DIR)
        sys.exit(1)
    video = sorted(mp4s, key=lambda p: p.stat().st_size, reverse=True)[0]
    print(f"    Downloaded → {video.name}")
    return video


def latest_download() -> Path:
    mp4s = list(DOWNLOADS_DIR.glob("*.mp4")) + list(DOWNLOADS_DIR.glob("*.mkv"))
    if not mp4s:
        print("ERROR: --skip-download used but no video found in downloads/")
        sys.exit(1)
    return max(mp4s, key=lambda p: p.stat().st_mtime)


# ── step 2: shot detection ─────────────────────────────────────────────────────

def detect_shots(video: Path, threshold: float = 0.3) -> list[dict]:
    print(f"\n[2/8] Detecting scene cuts (threshold={threshold})")
    duration = video_duration(video)

    # Use ffmpeg to write scene scores to stderr, parse timestamps from output
    r = subprocess.run(
        ["ffmpeg", "-i", str(video),
         "-vf", f"select=gt(scene\\,{threshold}),showinfo",
         "-vsync", "vfr", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    cut_times = []
    for line in r.stderr.splitlines():
        if "showinfo" in line and "pts_time:" in line:
            try:
                ts = float(line.split("pts_time:")[1].split()[0])
                cut_times.append(ts)
            except (IndexError, ValueError):
                pass

    cut_times = [0.0] + sorted(set(cut_times)) + [duration]

    shots = []
    for i in range(len(cut_times) - 1):
        start = cut_times[i]
        end = cut_times[i + 1]
        shots.append({"index": i, "start": round(start, 3), "end": round(end, 3),
                       "duration": round(end - start, 3)})

    fallback = False
    if len(shots) < 2 or len(shots) > 30:
        print(f"    WARNING: scene detection found {len(shots)} shot(s) — falling back to even splits")
        fallback = True
        n = 8
        chunk = duration / n
        shots = [{"index": i, "start": round(i * chunk, 3),
                   "end": round((i + 1) * chunk, 3),
                   "duration": round(chunk, 3)} for i in range(n)]

    for s in shots:
        flag = " ⚠ fallback" if fallback else ""
        print(f"    Shot {s['index']:02d}: {s['start']:.2f}s → {s['end']:.2f}s  ({s['duration']:.2f}s){flag}")
    return shots


# ── step 3: audio extraction ───────────────────────────────────────────────────

def extract_audio(video: Path) -> Path:
    print("\n[3/8] Extracting audio track")
    audio_out = TMP_DIR / (video.stem + "_audio.m4a")
    run(["ffmpeg", "-y", "-i", str(video), "-vn", "-acodec", "aac",
         "-b:a", "192k", str(audio_out)], capture=True)
    print(f"    Audio → {audio_out.name}")
    return audio_out


# ── step 4: describe trend shots ───────────────────────────────────────────────

def describe_trend_shots(client: anthropic.Anthropic, video: Path, shots: list[dict]) -> list[dict]:
    print(f"\n[4/8] Describing {len(shots)} trend shots with Claude vision")
    trend_shots_path = TMP_DIR / "trend_shots.json"

    described = []
    for shot in shots:
        mid = (shot["start"] + shot["end"]) / 2
        ts_list = [shot["start"] + (shot["duration"] * t) for t in (0.2, 0.5, 0.8)]
        ts_list = [min(max(t, 0), shot["end"] - 0.05) for t in ts_list]

        frame_dir = TMP_DIR / f"trend_shot_{shot['index']:02d}"
        frame_dir.mkdir(exist_ok=True)
        frames = extract_frames(video, ts_list, frame_dir)

        if not frames:
            print(f"    Shot {shot['index']:02d}: no frames extracted, skipping vision")
            desc = {"shot_type": "unknown", "energy_level": "medium", "setting": "unknown",
                    "subject": "unknown", "notable_action": "none",
                    "duration_seconds": shot["duration"]}
        else:
            desc = describe_shot(client, frames, shot["duration"])
            print(f"    Shot {shot['index']:02d}: {desc.get('shot_type','?')} — {desc.get('notable_action','?')}")

        described.append({**shot, "description": desc})

    trend_shots_path.write_text(json.dumps(described, indent=2))
    print(f"    Saved → tmp/trend_shots.json")
    return described


# ── step 5: index camera roll ──────────────────────────────────────────────────

def index_clips(client: anthropic.Anthropic, clips_dir: Path, force: bool = False) -> dict:
    print(f"\n[5/8] Indexing camera roll clips in {clips_dir}")
    index: dict = {}
    if CLIP_INDEX_PATH.exists() and not force:
        index = json.loads(CLIP_INDEX_PATH.read_text())
        print(f"    Loaded existing index with {len(index)} clips")

    clip_files = sorted(
        p for p in clips_dir.rglob("*")
        if p.suffix.lower() in (".mp4", ".mov", ".m4v", ".avi", ".mkv")
    )
    print(f"    Found {len(clip_files)} clip(s)")

    BATCH_SIZE = 10
    pending = []  # list of (clip, key, dur, frame_path)

    def flush_batch():
        if not pending:
            return
        names = [p[0].name for p in pending]
        print(f"    [tagging batch] {', '.join(names)}")
        content = []
        for clip, key, dur, frame_path in pending:
            content.append({"type": "text", "text": f"=== Clip: {clip.name} ({dur:.1f}s) ==="})
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": encode_image(frame_path)},
            })
        content.append({
            "type": "text",
            "text": (
                "For each clip above (identified by its === Clip: name === header), "
                "output a JSON array where each element has:\n"
                '{"filename":"...","shot_type":"...","energy_level":"low|medium|high",'
                '"setting":"...","subject":"...","notable_action":"...","duration_seconds":<float>}\n'
                "shot_type examples: wide establishing, medium, close-up, aerial, POV, selfie, b-roll\n"
                f"There are {len(pending)} clips. Output ONLY the JSON array, no markdown fences."
            ),
        })
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=150 * len(pending),
                messages=[{"role": "user", "content": content}],
            )
            text = msg.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            results = json.loads(text.strip())
            for i, (clip, key, dur, frame_path) in enumerate(pending):
                try:
                    r = results[i] if i < len(results) else {}
                    desc = {
                        "shot_type": r.get("shot_type", "unknown"),
                        "energy_level": r.get("energy_level", "medium"),
                        "setting": r.get("setting", "unknown"),
                        "subject": r.get("subject", "unknown"),
                        "notable_action": r.get("notable_action", "none"),
                        "duration_seconds": dur,
                    }
                    index[key] = {"filename": clip.name, "path": str(clip),
                                  "duration": dur, "hash": key.split("|")[1], "description": desc}
                except Exception as e2:
                    print(f"    WARNING: parse failed for {clip.name}: {e2}")
            CLIP_INDEX_PATH.write_text(json.dumps(index, indent=2))
        except Exception as e:
            print(f"    WARNING: batch failed: {e}")
        pending.clear()

    for clip in clip_files:
        try:
            clip_hash = file_hash(clip)
        except OSError:
            print(f"    [skipped] {clip.name} (not fully downloaded from iCloud)")
            continue
        key = clip.name + "|" + clip_hash
        if key in index:
            continue

        try:
            dur = video_duration(clip)
            frame_dir = TMP_DIR / f"clip_{clip.stem[:20]}"
            frame_dir.mkdir(exist_ok=True)
            frames = extract_frames(clip, [dur * 0.5], frame_dir)
            if not frames:
                raise ValueError("no frames extracted")
            pending.append((clip, key, dur, frames[0]))
            if len(pending) >= BATCH_SIZE:
                flush_batch()
        except Exception as e:
            print(f"    WARNING: failed to prepare {clip.name}: {e}")

    flush_batch()

    CLIP_INDEX_PATH.write_text(json.dumps(index, indent=2))
    print(f"    Index saved → clip_index.json ({len(index)} clips)")
    return index


# ── step 6: match shots to clips ──────────────────────────────────────────────

def match_shots(client: anthropic.Anthropic, trend_shots: list[dict], clip_index: dict) -> list[dict]:
    print(f"\n[6/8] Matching {len(trend_shots)} trend shots to clips")
    clips_list = list(clip_index.values())

    clips_summary = "\n".join(
        f"{i}: {c['filename']} ({c['duration']:.1f}s) — {json.dumps(c['description'])}"
        for i, c in enumerate(clips_list)
    )

    matches = []
    for shot in trend_shots:
        prompt = (
            f"I need to recreate a video slot.\n\n"
            f"Slot duration: {shot['duration']:.2f}s\n"
            f"Slot description: {json.dumps(shot['description'])}\n\n"
            f"Available clips:\n{clips_summary}\n\n"
            f"Rank the top 3 best-matching clips by index. Reply with ONLY a JSON array of 3 objects:\n"
            f'[{{"index": <int>, "filename": "...", "reason": "one-line reason"}}, ...]\n'
            f"Prioritize matching shot_type, energy_level, and setting."
        )
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        ranking = json.loads(text.strip())

        best = ranking[0]
        clip = clips_list[best["index"]]
        print(f"    Slot {shot['index']:02d} → {clip['filename']}  ({best['reason']})")
        matches.append({
            "slot": shot,
            "best": {**best, "clip": clip},
            "ranking": ranking,
        })
    return matches


# ── step 7: assembly ───────────────────────────────────────────────────────────

def trim_clip(clip_path: Path, duration: float, out_path: Path) -> tuple[Path, str | None]:
    """Trim clip to duration (center crop). Returns (path, warning|None)."""
    clip_dur = video_duration(clip_path)
    warning = None
    if clip_dur < duration:
        warning = f"clip is {clip_dur:.1f}s but slot needs {duration:.1f}s — using full clip"
        start = 0.0
        trim_dur = clip_dur
    else:
        start = max(0.0, (clip_dur - duration) / 2)
        trim_dur = duration

    run(
        ["ffmpeg", "-y", "-ss", str(start), "-i", str(clip_path),
         "-t", str(trim_dur), "-c:v", "libx264", "-preset", "fast",
         "-crf", "23", "-an", str(out_path)],
        capture=True,
    )
    return out_path, warning


def assemble(matches: list[dict], audio_path: Path, output_path: Path) -> list[str]:
    print(f"\n[7/8] Assembling output video")
    trimmed_clips = []
    warnings = []

    for m in matches:
        slot = m["slot"]
        clip_info = m["best"]["clip"]
        clip_path = Path(clip_info["path"])
        out_trim = TMP_DIR / f"trimmed_{slot['index']:02d}.mp4"

        if not clip_path.exists():
            warnings.append(f"Slot {slot['index']}: clip file not found: {clip_path}")
            continue

        _, warn = trim_clip(clip_path, slot["duration"], out_trim)
        if warn:
            warnings.append(f"Slot {slot['index']}: {warn}")
        trimmed_clips.append(out_trim)
        print(f"    Trimmed slot {slot['index']:02d}: {clip_info['filename']}")

    if not trimmed_clips:
        print("ERROR: no clips to assemble")
        sys.exit(1)

    # write concat list
    concat_list = TMP_DIR / "concat.txt"
    concat_list.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in trimmed_clips)
    )

    # concatenate
    silent_out = TMP_DIR / "silent_concat.mp4"
    run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", str(concat_list), "-c:v", "libx264", "-preset", "fast",
         "-crf", "23", str(silent_out)],
        capture=True,
    )

    # overlay audio (loop if needed, trim to video length)
    video_dur = video_duration(silent_out)
    run(
        ["ffmpeg", "-y",
         "-i", str(silent_out),
         "-stream_loop", "-1", "-i", str(audio_path),
         "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-t", str(video_dur), "-shortest",
         str(output_path)],
        capture=True,
    )
    print(f"    Output → {output_path}")
    return warnings


# ── step 8: summary ────────────────────────────────────────────────────────────

def print_summary(matches: list[dict], warnings: list[str]):
    print("\n" + "═" * 60)
    print("SLOT SUMMARY")
    print("═" * 60)
    for m in matches:
        slot = m["slot"]
        best = m["best"]
        clip = best["clip"]
        print(f"\nSlot {slot['index']:02d}  {slot['start']:.2f}s–{slot['end']:.2f}s  ({slot['duration']:.2f}s)")
        desc = slot["description"]
        print(f"  Trend:  {desc.get('shot_type','?')} | {desc.get('energy_level','?')} energy | "
              f"{desc.get('setting','?')} | {desc.get('notable_action','?')}")
        print(f"  Match:  {clip['filename']}")
        print(f"  Reason: {best['reason']}")
        print(f"  Alt 1:  {m['ranking'][1]['filename'] if len(m['ranking']) > 1 else 'N/A'} — "
              f"{m['ranking'][1]['reason'] if len(m['ranking']) > 1 else ''}")
        print(f"  Alt 2:  {m['ranking'][2]['filename'] if len(m['ranking']) > 2 else 'N/A'} — "
              f"{m['ranking'][2]['reason'] if len(m['ranking']) > 2 else ''}")

    if warnings:
        print("\n⚠  WARNINGS:")
        for w in warnings:
            print(f"   • {w}")
    print("\n" + "═" * 60)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Recreate an Instagram Reel with your own clips")
    parser.add_argument("url", nargs="?", help="Instagram Reel URL")
    parser.add_argument("--clips", required=True, help="Folder of your camera roll clips")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Scene-cut threshold 0.0–1.0 (default 0.3)")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "recreated_reel.mp4"))
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, reuse latest file in downloads/")
    parser.add_argument("--force-reindex", action="store_true",
                        help="Re-tag all camera roll clips, ignoring cache")
    parser.add_argument("--cookies-from-browser", metavar="BROWSER",
                        help="Pass cookies from browser to yt-dlp (e.g. chrome, firefox, edge)")
    parser.add_argument("--cookies", metavar="FILE",
                        help="Path to cookies.txt file exported from your browser")
    args = parser.parse_args()

    check_deps()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    clips_dir = Path(args.clips)
    if not clips_dir.is_dir():
        print(f"ERROR: clips folder not found: {clips_dir}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1
    if args.skip_download:
        video = latest_download()
        print(f"\n[1/8] Using existing download: {video.name}")
    else:
        if not args.url:
            parser.error("URL is required unless --skip-download is used")
        video = download_reel(args.url, cookies_from_browser=args.cookies_from_browser, cookies=args.cookies)

    # Step 2
    shots = detect_shots(video, args.threshold)

    # Step 3
    audio = extract_audio(video)

    # Step 4
    trend_shots = describe_trend_shots(client, video, shots)

    # Step 5
    clip_index = index_clips(client, clips_dir, force=args.force_reindex)

    if not clip_index:
        print("ERROR: no clips indexed. Check your --clips folder for .mp4/.mov files.")
        sys.exit(1)

    # Step 6
    matches = match_shots(client, trend_shots, clip_index)

    # Step 7
    warnings = assemble(matches, audio, output_path)

    # Step 8
    print_summary(matches, warnings)
    print(f"\nDone! Output: {output_path}")


if __name__ == "__main__":
    main()
