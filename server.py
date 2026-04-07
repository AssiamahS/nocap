#!/usr/bin/env python3
"""nocap web server - Video rep counter viewer with timeline scrubbing."""

import json
import os
import re
import shutil
import uuid
from pathlib import Path
from flask import Flask, send_from_directory, jsonify, request, send_file
import subprocess
import sys

app = Flask(__name__, static_folder="web")

DATA_DIR = Path(os.environ.get("NOCAP_DATA", Path.home() / "nocap-data"))
RESULTS_DIR = DATA_DIR / "results"
VIDEOS_DIR = Path(os.environ.get("NOCAP_VIDEOS", Path.home() / "Downloads"))
WEB_DIR = Path(__file__).parent / "web"
DOWNLOADS_DIR = Path(__file__).parent / "videos"
DOWNLOADS_DIR.mkdir(exist_ok=True)


@app.route("/")
def index():
    return send_from_directory("web", "index.html")


@app.route("/web/<path:filename>")
def static_files(filename):
    return send_from_directory("web", filename)


@app.route("/api/sessions")
def list_sessions():
    sessions = []
    if RESULTS_DIR.exists():
        for f in sorted(RESULTS_DIR.glob("*.json")):
            with open(f) as fh:
                data = json.load(fh)
                data["id"] = f.stem
                sessions.append(data)
    return jsonify(sessions)


@app.route("/api/session/<session_id>")
def get_session(session_id):
    path = RESULTS_DIR / f"{session_id}.json"
    if not path.exists():
        return jsonify({"error": "not found"}), 404
    with open(path) as f:
        data = json.load(f)
        data["id"] = session_id
    return jsonify(data)


@app.route("/api/video/<path:filename>")
def serve_video(filename):
    for base in [DOWNLOADS_DIR, VIDEOS_DIR, DATA_DIR / "videos", Path.home() / "Downloads"]:
        path = base / filename
        if path.exists():
            return send_file(path, conditional=True)
    return jsonify({"error": "video not found"}), 404


@app.route("/api/annotated/<session_id>")
def serve_annotated(session_id):
    """Serve annotated skeleton video. Files are named {session_id}_annotated.mp4."""
    path = WEB_DIR / f"{session_id}_annotated.mp4"
    if path.exists():
        return send_file(path, conditional=True)
    return jsonify({"error": "annotated video not found"}), 404


@app.route("/api/poses/<session_id>")
def serve_poses(session_id):
    """Serve pose data. Files are named {session_id}_poses.json."""
    path = WEB_DIR / f"{session_id}_poses.json"
    if path.exists():
        return send_file(path)
    return jsonify({"error": "pose data not found"}), 404


@app.route("/api/session/<session_id>/rename", methods=["POST"])
def rename_session(session_id):
    """Rename a session (update its display title)."""
    data = request.json
    new_name = data.get("name", "").strip()
    if not new_name:
        return jsonify({"error": "name required"}), 400

    path = RESULTS_DIR / f"{session_id}.json"
    if not path.exists():
        return jsonify({"error": "session not found"}), 404

    with open(path) as f:
        session = json.load(f)

    session["display_name"] = new_name
    with open(path, "w") as f:
        json.dump(session, f, indent=2)

    return jsonify({"ok": True, "name": new_name})


@app.route("/api/session/<session_id>/exercise", methods=["POST"])
def update_exercise(session_id):
    """Update the exercise type for a session."""
    data = request.json
    exercise = data.get("exercise", "").strip()
    if not exercise:
        return jsonify({"error": "exercise required"}), 400

    path = RESULTS_DIR / f"{session_id}.json"
    if not path.exists():
        return jsonify({"error": "session not found"}), 404

    with open(path) as f:
        session = json.load(f)

    session["exercise"] = exercise
    with open(path, "w") as f:
        json.dump(session, f, indent=2)

    return jsonify({"ok": True, "exercise": exercise})


@app.route("/api/analyze", methods=["POST"])
def analyze_video():
    """Run rep counter on a video file or URL."""
    data = request.json
    url = data.get("url")
    video_path = data.get("video_path")
    exercise = data.get("exercise", "bench_press")

    # Download from URL if provided
    if url:
        try:
            result = download_video(url)
            video_path = result["path"]
            title = result["title"]
        except Exception as e:
            return jsonify({"error": f"Download failed: {str(e)}"}), 400
    elif not video_path or not Path(video_path).exists():
        return jsonify({"error": "No video_path or url provided"}), 400
    else:
        title = Path(video_path).stem

    # Make a safe session ID
    safe_title = re.sub(r'[^a-zA-Z0-9_-]', '_', title)[:50]
    session_id = f"{safe_title}_{exercise}"

    # Run rep counter with annotated output + pose data
    annotated_path = str(WEB_DIR / f"{session_id}_annotated.mp4")
    poses_path = str(WEB_DIR / f"{session_id}_poses.json")

    proc = subprocess.run(
        [sys.executable, "count_reps.py", video_path, "-e", exercise,
         "-o", annotated_path, "-p", poses_path, "--json"],
        capture_output=True, text=True, cwd=Path(__file__).parent,
        timeout=600,
    )

    if proc.returncode != 0:
        return jsonify({"error": proc.stderr[-500:]}), 500

    # Parse JSON result
    lines = proc.stdout.strip().split("\n")
    json_start = next((i for i, l in enumerate(lines) if l.strip().startswith("{")), None)
    if json_start is None:
        return jsonify({"error": "No JSON output from counter"}), 500

    json_str = "\n".join(lines[json_start:])
    result = json.loads(json_str)

    # Copy video to local videos dir if it's not already there
    local_video = DOWNLOADS_DIR / Path(video_path).name
    if not local_video.exists():
        shutil.copy2(video_path, local_video)

    # Save session results
    session_data = {
        "video": Path(video_path).name,
        "exercise": exercise,
        "reps_counted": result["reps"],
        "reps_actual": None,
        "rep_timestamps_sec": result["rep_timestamps_sec"],
        "duration_sec": result["duration_sec"],
        "fps": result["fps"],
        "total_frames": result["total_frames"],
        "pose_detection_rate": result["pose_detection_rate"],
        "primary_side": result["primary_side"],
        "angle_range_deg": result["angle_range_deg"],
        "exercise_start_sec": result.get("exercise_start_sec", 0),
        "date": "2026-04-07",
        "source": url or video_path,
    }

    results_path = RESULTS_DIR / f"{session_id}.json"
    with open(results_path, "w") as f:
        json.dump(session_data, f, indent=2)

    session_data["id"] = session_id
    return jsonify(session_data)


def download_video(url):
    """Download video from URL using yt-dlp."""
    output_template = str(DOWNLOADS_DIR / "%(title).50s_%(id)s.%(ext)s")

    # Try with cookies from browser for sites that need auth
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f", "best[height<=720]/best",  # Fallback if 720p not available
        "-o", output_template,
        "--print", "after_move:filepath",
        "--print", "%(title)s",
        "--no-overwrites",
        "--restrict-filenames",
        url,
    ]

    print(f"[nocap] Downloading: {url}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

    print(f"[nocap] yt-dlp exit code: {proc.returncode}")
    if proc.stderr:
        print(f"[nocap] yt-dlp stderr: {proc.stderr[-500:]}")
    if proc.stdout:
        print(f"[nocap] yt-dlp stdout: {proc.stdout[-500:]}")

    if proc.returncode != 0:
        err = proc.stderr.strip()
        # Common errors with helpful messages
        if "HTTP Error 403" in err or "HTTP Error 429" in err:
            raise Exception("Site blocked the download. Try a different URL or shorter video.")
        if "Unsupported URL" in err:
            raise Exception(f"URL not supported. Try YouTube, TikTok, or Instagram.")
        if "Video unavailable" in err or "Private video" in err:
            raise Exception("Video is private or unavailable.")
        raise Exception(err[-300:])

    lines = proc.stdout.strip().split("\n")
    # yt-dlp --print outputs: filepath on one line, title on next
    # But it also prints download progress lines, so take the last 2 non-empty
    non_empty = [l.strip() for l in lines if l.strip()]

    filepath = None
    title = "unknown"

    # Find the filepath (it's an absolute path or relative path that exists)
    for line in reversed(non_empty):
        if Path(line).exists():
            filepath = line
            break

    if not filepath:
        # Look for most recently modified file in downloads dir
        files = sorted(DOWNLOADS_DIR.glob("*"), key=lambda f: f.stat().st_mtime, reverse=True)
        video_files = [f for f in files if f.suffix in ('.mp4', '.webm', '.mkv', '.mov')]
        if video_files:
            filepath = str(video_files[0])
        else:
            raise Exception("Download completed but can't find the file")

    # Title is the last line
    title = non_empty[-1] if non_empty else Path(filepath).stem
    if Path(title).exists():
        # It's a filepath, not a title
        title = Path(filepath).stem

    print(f"[nocap] Downloaded: {filepath} (title: {title})")
    return {"path": filepath, "title": title}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8765))
    print(f"nocap server running on http://localhost:{port}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Videos dir: {VIDEOS_DIR}")
    app.run(host="0.0.0.0", port=port, debug=True)
