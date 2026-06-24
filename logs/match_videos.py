#!/usr/bin/env python
"""
match_videos.py — Match iPhone MOV/MP4 recordings to trial summary logs
                  and rename them to the canonical trial name.

Usage
-----
    python logs/match_videos.py                   # dry-run: print matches only
    python logs/match_videos.py --apply           # rename videos
    python logs/match_videos.py --tz-offset 2     # UTC offset of PC clock (default: 2 = Italy CEST)
    python logs/match_videos.py --video-dir path  # custom video folder (default: logs/video/)

Workflow
--------
1. Copy iPhone videos into  logs/video/  (keep the original IMG_XXXX.MOV names).
2. Run this script with --apply.
   Each video is renamed to  trial_P001_T05_B_with_context.MOV
3. Open BORIS, add a new observation for each renamed video.
   BORIS pre-fills the observation ID with the filename (without extension),
   e.g. trial_P001_T05_B_with_context — which is exactly what analyze_results.py
   expects. No mapping file, no manual editing needed.
4. Annotate the non_assembly behavior.
5. Export BORIS events as CSV into the same video dir.
6. Run: python logs/analyze_results.py --boris-dir logs/video

How matching works
------------------
iPhone MOV files embed the recording start time in the QuickTime container
(mvhd atom, UTC). This timestamp survives file transfers.
Trial summary JSONs store timestamp_start in local time.
The script converts both to UTC and accepts a match when the gap is < --time-tol
(default 120 s). Video duration vs atct_seconds is shown as a secondary check.
"""

import argparse
import datetime
import json
import re
import struct
from pathlib import Path

# Pattern that already-renamed videos satisfy — no need to process them again.
_RENAMED_RE = re.compile(r"^trial_[^_]+_T\d+_.+$", re.IGNORECASE)


MAC_EPOCH = datetime.datetime(1904, 1, 1, tzinfo=datetime.timezone.utc)


# ── QuickTime metadata ────────────────────────────────────────────────────────

def _read_mvhd(path: Path) -> dict | None:
    """
    Extract creation_utc and duration_s from the QuickTime mvhd atom.
    iPhone finalises the file by writing moov at the END, so we scan the tail.
    """
    size = path.stat().st_size
    tail_size = min(20 * 1024 * 1024, size)

    with open(path, "rb") as f:
        f.seek(size - tail_size)
        tail = f.read()

    idx = tail.rfind(b"mvhd")
    if idx == -1:
        with open(path, "rb") as f:
            head = f.read(min(20 * 1024 * 1024, size))
        idx = head.find(b"mvhd")
        if idx == -1:
            return None
        atom = head[idx + 4:]
    else:
        atom = tail[idx + 4:]

    version = atom[0]
    try:
        if version == 0:
            ctime_s   = struct.unpack(">I", atom[4:8])[0]
            timescale = struct.unpack(">I", atom[12:16])[0]
            duration  = struct.unpack(">I", atom[16:20])[0]
        else:
            ctime_s   = struct.unpack(">Q", atom[4:12])[0]
            timescale = struct.unpack(">I", atom[20:24])[0]
            duration  = struct.unpack(">Q", atom[24:32])[0]
    except struct.error:
        return None

    return {
        "creation_utc": MAC_EPOCH + datetime.timedelta(seconds=int(ctime_s)),
        "duration_s":   duration / timescale if timescale else 0.0,
    }


# ── trial loader ──────────────────────────────────────────────────────────────

def load_trials(log_dir: Path, tz_offset: float) -> list[dict]:
    tz = datetime.timezone(datetime.timedelta(hours=tz_offset))
    trials = []
    for sf in sorted(log_dir.glob("*_summary.json")):
        with open(sf, encoding="utf-8") as f:
            s = json.load(f)
        raw = s.get("timestamp_start")
        if not raw:
            continue
        local_dt  = datetime.datetime.fromisoformat(raw).replace(tzinfo=tz)
        start_utc = local_dt.astimezone(datetime.timezone.utc)
        # In the summary JSON the field "trial_id" actually stores the participant id
        # (e.g. "P001") and "trial_number" stores the trial index.
        trials.append({
            "summary_file":   sf.name,
            "participant_id": s.get("trial_id", "?"),
            "trial_number":   s.get("trial_number", 0),
            "condition":      s.get("condition", "unknown"),
            "start_utc":      start_utc,
            "atct_seconds":   s.get("atct_seconds"),
        })
    return trials


def _trial_basename(trial: dict) -> str:
    """Return the canonical base name for a trial (no extension)."""
    pid  = trial["participant_id"]
    tnum = int(trial["trial_number"])
    cond = trial["condition"]
    return f"trial_{pid}_T{tnum:02d}_{cond}"


# ── matching ──────────────────────────────────────────────────────────────────

def match_video(video_path: Path, trials: list[dict],
                time_tol_s: float = 120.0) -> list[dict]:
    meta = _read_mvhd(video_path)
    if meta is None:
        return []

    vid_start = meta["creation_utc"]
    vid_dur   = meta["duration_s"]
    results   = []
    for t in trials:
        gap_s = abs((vid_start - t["start_utc"]).total_seconds())
        if gap_s > time_tol_s:
            continue
        dur_diff = abs(vid_dur - t["atct_seconds"]) if t["atct_seconds"] and vid_dur > 0 else None
        results.append({**t, "video_start_utc": vid_start,
                        "video_duration_s": vid_dur,
                        "time_gap_s": gap_s, "duration_diff_s": dur_diff})

    results.sort(key=lambda x: x["time_gap_s"])
    return results


# ── interactive rename for unmatched videos ───────────────────────────────────

def _prompt_rename(vp: Path) -> Path | None:
    """
    Interactively ask the user for participant ID, trial number, and condition,
    then rename the video. Returns the new Path, or None if the user skips.
    """
    print()
    print(f"  Enter trial info for '{vp.name}' (or press Enter to skip):")

    pid = input("    Participant ID  [e.g. P001]: ").strip()
    if not pid:
        print("    Skipped.")
        return None

    tnum_raw = input("    Trial number    [e.g. 1]:    ").strip()
    if not tnum_raw:
        print("    Skipped.")
        return None
    try:
        tnum = int(tnum_raw)
    except ValueError:
        print("    Invalid number — skipped.")
        return None

    print("    Condition options:")
    print("      1) A_no_robot")
    print("      2) B_no_context")
    print("      3) B_with_context")
    cond_choice = input("    Choice [1/2/3]: ").strip()
    cond_map = {"1": "A_no_robot", "2": "B_no_context", "3": "B_with_context"}
    condition = cond_map.get(cond_choice)
    if condition is None:
        print("    Invalid choice — skipped.")
        return None

    basename = f"trial_{pid}_T{tnum:02d}_{condition}"
    new_name = basename + vp.suffix
    new_path = vp.parent / new_name

    print(f"    Rename to: {new_name}")
    confirm = input("    Confirm? [y/n]: ").strip().lower()
    if confirm != "y":
        print("    Skipped.")
        return None

    if new_path.exists():
        print(f"    WARNING: {new_name} already exists — skipped.")
        return None

    vp.rename(new_path)
    print(f"    Renamed OK")
    return new_path


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Match iPhone videos to trial logs, rename them, update boris_mapping.json."
    )
    parser.add_argument("--video-dir", default=None,
                        help="Folder containing .MOV/.MP4 files "
                             "(default: logs/video/ next to this script)")
    parser.add_argument("--log-dir",   default=None,
                        help="Folder with *_summary.json files "
                             "(default: same folder as this script)")
    parser.add_argument("--tz-offset", type=float, default=2.0,
                        help="UTC offset of the PC clock in hours "
                             "(default: 2 = Italy CEST)")
    parser.add_argument("--time-tol",  type=float, default=120.0,
                        help="Max time gap (s) to accept a match (default: 120)")
    parser.add_argument("--apply",     action="store_true",
                        help="Rename videos and write boris_mapping.json "
                             "(without this flag the script is a dry-run)")
    args = parser.parse_args()

    here      = Path(__file__).parent.resolve()          # logs/
    log_dir   = Path(args.log_dir)   if args.log_dir   else here
    video_dir = Path(args.video_dir) if args.video_dir else here / "video"

    if not video_dir.exists():
        print(f"Video directory not found: {video_dir}")
        return

    videos = sorted(p for p in video_dir.iterdir()
                    if p.suffix.lower() in {".mov", ".mp4", ".m4v"})
    if not videos:
        print(f"No video files found in {video_dir}")
        return

    trials = load_trials(log_dir, args.tz_offset)
    if not trials:
        print(f"No *_summary.json files found in {log_dir}")
        return

    mode = "APPLY" if args.apply else "DRY-RUN (pass --apply to rename)"
    print(f"\n[{mode}]")
    print(f"  Videos   : {video_dir}  ({len(videos)} file(s))")
    print(f"  Logs     : {log_dir}  ({len(trials)} trial(s))")
    print(f"  TZ offset: +{args.tz_offset:.1f} h  |  tolerance: {args.time_tol:.0f} s\n")
    print("=" * 70)

    unmatched = []   # videos with no trial log match (expected for A_no_robot)

    for vp in videos:
        if _RENAMED_RE.match(vp.stem):
            print(f"\n[{vp.name}]  already renamed — skipping")
            continue

        meta = _read_mvhd(vp)
        if meta is None:
            print(f"\n[{vp.name}]  WARNING: cannot read QuickTime metadata — skipping")
            continue

        local_time = (meta["creation_utc"] + datetime.timedelta(hours=args.tz_offset)).strftime("%H:%M:%S")
        print(f"\n[{vp.name}]")
        print(f"  Recorded : {meta['creation_utc'].strftime('%Y-%m-%d %H:%M:%S')} UTC  (= {local_time} local)")
        print(f"  Duration : {meta['duration_s']:.1f} s  ({meta['duration_s']/60:.1f} min)")

        matches = match_video(vp, trials, args.time_tol)

        if not matches:
            print("  No trial log matched — likely a condition A (no robot) video.")
            if args.apply:
                new_path = _prompt_rename(vp)
                if new_path is None:
                    unmatched.append(vp)
                # else: already renamed inside _prompt_rename
            else:
                print("  Run with --apply to be prompted for the trial info.")
                unmatched.append(vp)
            continue

        # Accept automatically only if there is one clear best match
        if len(matches) > 1 and matches[0]["time_gap_s"] >= 10:
            print("  Multiple candidates — pick manually:")
            for i, m in enumerate(matches[:3]):
                extra = f"  dur_diff={m['duration_diff_s']:.1f}s" if m["duration_diff_s"] is not None else ""
                print(f"    [{i+1}] {m['summary_file']}  gap={m['time_gap_s']:.1f}s{extra}")
            unmatched.append(vp)
            continue

        m        = matches[0]
        basename = _trial_basename(m)
        new_name = basename + vp.suffix   # e.g. trial_P001_T05_B_with_context.MOV
        new_path = vp.parent / new_name
        verdict  = "MATCH" if m["time_gap_s"] < 30 else "POSSIBLE MATCH"

        print(f"  {verdict}: {m['summary_file']}")
        print(f"    trial_tag  : T{int(m['trial_number']):02d}")
        print(f"    condition  : {m['condition']}")
        print(f"    time gap   : {m['time_gap_s']:.1f} s")
        if m["duration_diff_s"] is not None:
            print(f"    dur diff   : {m['duration_diff_s']:.1f} s  "
                  f"(video={m['video_duration_s']:.1f}s  atct={m['atct_seconds']:.1f}s)")
        print(f"    rename to  : {new_name}")

        if vp.name == new_name:
            print("    (already correctly named)")
            continue

        if args.apply:
            if new_path.exists():
                print(f"    WARNING: {new_name} already exists — skipping rename")
            else:
                vp.rename(new_path)
                print(f"    Renamed OK")

    print("\n" + "=" * 70)

    if unmatched:
        print(f"\nSkipped videos ({len(unmatched)}) — run with --apply to be prompted:")
        for vp in unmatched:
            print(f"  {vp.name}")

    if args.apply:
        print("\nNext steps:")
        print(f"  1. Open BORIS, add a new Observation for each renamed video.")
        print(f"     BORIS pre-fills the Observation ID with the filename (no extension).")
        print(f"     e.g.  trial_P001_T05_B_with_context  — do not change it.")
        print(f"  2. Annotate the 'non_assembly' behavior.")
        print(f"  3. Export events as CSV into: {video_dir}")
        print(f"  4. Run: python logs/analyze_results.py --boris-dir {video_dir}")
    else:
        print(f"\nDry-run complete. Run with --apply to rename files interactively.")


if __name__ == "__main__":
    main()
