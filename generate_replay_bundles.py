#!/usr/bin/env python3
"""Generate compact replay bundles from TELLAgent run logs for the report viewer.

Reads action_frames.jsonl + messages.jsonl from each game's replay directory,
produces delta-compressed JSON bundles + a thumbnails file.

Usage:
    python3 generate_replay_bundles.py
"""

import json
import os
import sys
from pathlib import Path

REPLAY_BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("REPLAY_BASE", "run_logs/replays"))
OUTPUT_DIR = Path(__file__).parent / "replay_data"

# Map game IDs to directory names (handle both naming conventions)
def find_game_dirs():
    dirs = {}
    for d in sorted(REPLAY_BASE.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        # Extract game ID from directory name
        # Formats:
        #   <agent>-<config_name>-<game_id>-YYYYMMDD-...
        #   run_<game_id>-...
        import re
        if re.match(r"[a-z_]+-.*-\d{8}-", name):
            # Find game_id: it's the 2-6 char code before the YYYYMMDD date
            m = re.search(r"-([a-z0-9]{2,6})-\d{8}-", name)
            game_id = m.group(1) if m else None
            if not game_id:
                continue
        elif name.startswith("run_"):
            game_id = name.split("-")[0].replace("run_", "")
        else:
            continue
        dirs[game_id] = d
    return dirs


def read_jsonl(path):
    lines = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    return lines


def compute_delta(prev_grid, curr_grid):
    """Return list of [x, y, value] for changed cells."""
    delta = []
    for y in range(len(curr_grid)):
        for x in range(len(curr_grid[y])):
            if prev_grid[y][x] != curr_grid[y][x]:
                delta.append([x, y, curr_grid[y][x]])
    return delta


def format_action(action_obj):
    """Format action dict as a readable string."""
    if not action_obj:
        return ""
    name = action_obj.get("name", "")
    args = action_obj.get("args", {})
    if name == "click" and "x" in args and "y" in args:
        return f"click({args['x']},{args['y']})"
    if args:
        return f"{name}({','.join(str(v) for v in args.values())})"
    return name


def process_action_frames(path, max_actions=None, max_frames=2048):
    """Read action_frames.jsonl and return delta-compressed frames + thumbnail + seq mapping.

    Includes ALL animation frames per action (not just the last one),
    so the replay shows smooth transitions.

    If max_actions is set, truncate after that many action entries
    (used to stop at game win rather than including post-win actions).
    """
    raw = read_jsonl(path)
    if not raw:
        return [], None, []

    if max_actions is not None:
        raw = raw[:max_actions]

    frames = []
    prev_grid = None
    thumbnail = None
    # Store (frame_index, message_seq) mapping for later use
    frame_seqs = []

    for i, entry in enumerate(raw[:max_frames]):
        result = entry.get("result", {})
        obs = result.get("observation", {})
        obs_frames = obs.get("frames", [])
        action = entry.get("action", {})
        action_str = format_action(action)
        msg_seq = entry.get("seq", 0)

        if not obs_frames:
            continue

        # Record the mapping at the first animation frame of this action
        frame_seqs.append((len(frames), msg_seq))

        for fi, anim_frame in enumerate(obs_frames):
            grid = anim_frame.get("grid")
            if grid is None:
                continue

            if thumbnail is None:
                thumbnail = grid

            # Only show action label on the first animation frame
            label = action_str if fi == 0 else ""

            if prev_grid is None:
                frames.append({"s": i, "g": grid, "a": label})
            else:
                delta = compute_delta(prev_grid, grid)
                if delta:
                    frames.append({"s": i, "d": delta, "a": label})
                elif fi == 0:
                    # No visual change but record the action
                    frames.append({"s": i, "a": label})

            prev_grid = grid

    return frames, thumbnail, frame_seqs


def process_messages(path, max_thinking_chars=500):
    """Extract assistant thinking text and tool call summaries."""
    raw = read_jsonl(path)
    thinking = []

    for entry in raw:
        role = entry.get("role", "")
        if role != "assistant":
            continue

        parts = entry.get("parts", [])
        seq = entry.get("seq", 0)

        # Extract thinking text
        for part in parts:
            if part.get("thought") and part.get("text"):
                text = part["text"].strip()
                if len(text) > max_thinking_chars:
                    text = text[:max_thinking_chars] + "..."
                thinking.append({"s": seq, "t": text})
                break

        # Extract tool call name
        for part in parts:
            fc = part.get("functionCall")
            if fc:
                name = fc.get("name", "")
                args = fc.get("args", {})
                # For bash_exec, include a snippet of the command
                summary = name
                if name == "bash_exec" and "command" in args:
                    cmd = str(args["command"])[:100]
                    summary = f"bash_exec: {cmd}"
                elif name == "write_file" and "path" in args:
                    summary = f"write_file: {args['path']}"
                thinking.append({"s": seq, "c": summary})
                break

    return thinking


def extract_memory_snapshots(path):
    """Extract MEMORY.md content at each write_file call.

    Returns list of {seq, content} dicts.
    """
    raw = read_jsonl(path)
    snapshots = []

    for entry in raw:
        if entry.get("role") != "assistant":
            continue
        parts = entry.get("parts", [])
        seq = entry.get("seq", 0)

        for part in parts:
            fc = part.get("functionCall")
            if not fc:
                continue
            if fc.get("name") != "write_file":
                continue
            args = fc.get("args", {})
            fpath = str(args.get("path", ""))
            if "MEMORY" not in fpath.upper():
                continue
            content = args.get("content", "")
            if content:
                snapshots.append({"seq": seq, "content": content})
            break

    return snapshots


def process_game(game_id, game_dir):
    """Process a single game directory into a replay bundle."""
    af_path = game_dir / "action_frames.jsonl"
    msg_path = game_dir / "messages.jsonl"
    manifest_path = game_dir / "manifest.json"

    if not af_path.exists():
        print(f"  SKIP {game_id}: no action_frames.jsonl")
        return None, None

    # Read manifest
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    # Truncate at game win if manifest records total action count
    max_actions = manifest.get("action_frame_count")

    # Process frames (need frame_seqs for level step mapping)
    frames, thumbnail, frame_seqs = process_action_frames(af_path, max_actions=max_actions)

    # Build action-index (0-based) to first-frame-index mapping
    action_to_frame = {}
    for f_idx, frame in enumerate(frames):
        act_idx = frame.get("s", 0)
        if act_idx not in action_to_frame:
            action_to_frame[act_idx] = f_idx

    # Level completion steps — map manifest action counts to frame indices.
    # manifest action_frame_count is 1-based cumulative (e.g. 4 = 4th action),
    # action_to_frame keys are 0-based (0 = first action).
    level_steps = []
    for lc in manifest.get("level_completion_steps", []):
        action_count = lc.get("action_frame_count", 0)
        # Convert 1-based cumulative to 0-based index: the action that
        # completed the level is at index (action_count - 1).
        act_idx = action_count - 1
        if act_idx in action_to_frame:
            level_steps.append(action_to_frame[act_idx])
        elif action_to_frame:
            closest = min(action_to_frame.keys(), key=lambda a: abs(a - act_idx), default=0)
            level_steps.append(action_to_frame.get(closest, 0))

    # Process messages
    thinking = []
    if msg_path.exists():
        thinking = process_messages(msg_path)

    # Extract MEMORY.md snapshots with content
    raw_snapshots = []
    if msg_path.exists():
        raw_snapshots = extract_memory_snapshots(msg_path)

    # Map memory snapshots to frame indices
    memory_updates = []
    memory_snapshots = []
    if frame_seqs and raw_snapshots:
        for snap in raw_snapshots:
            msg_seq = snap["seq"]
            best_fi = 0
            best_dist = float('inf')
            for fi, ms in frame_seqs:
                dist = abs(ms - msg_seq)
                if dist < best_dist:
                    best_dist = dist
                    best_fi = fi
            memory_updates.append(best_fi)
            memory_snapshots.append({
                "frame": best_fi,
                "content": snap["content"],
            })

    bundle = {
        "game_id": game_id,
        "frame_count": len(frames),
        "level_steps": level_steps,
        "memory_updates": memory_updates,
        "memory_snapshots": memory_snapshots,
        "frames": frames,
        "thinking": thinking,
    }

    return bundle, thumbnail


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    game_dirs = find_game_dirs()
    print(f"Found {len(game_dirs)} games: {', '.join(sorted(game_dirs.keys()))}")

    thumbnails = {}
    total_size = 0

    for game_id in sorted(game_dirs.keys()):
        game_dir = game_dirs[game_id]
        print(f"Processing {game_id}...")

        bundle, thumbnail = process_game(game_id, game_dir)
        if bundle is None:
            continue

        # Save per-game bundle
        out_path = OUTPUT_DIR / f"{game_id}.json"
        with open(out_path, "w") as f:
            json.dump(bundle, f, separators=(",", ":"))

        size = out_path.stat().st_size
        total_size += size
        print(f"  {game_id}: {len(bundle['frames'])} frames, {len(bundle['thinking'])} thinking, {size/1024:.0f}KB")

        if thumbnail:
            thumbnails[game_id] = thumbnail

    # Save thumbnails
    thumb_path = OUTPUT_DIR / "thumbnails.json"
    with open(thumb_path, "w") as f:
        json.dump(thumbnails, f, separators=(",", ":"))

    thumb_size = thumb_path.stat().st_size
    print(f"\nDone! {len(thumbnails)} thumbnails ({thumb_size/1024:.0f}KB)")
    print(f"Total replay data: {total_size/1024/1024:.1f}MB")


if __name__ == "__main__":
    main()
