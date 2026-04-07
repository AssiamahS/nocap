#!/usr/bin/env python3
"""nocap web server - Video rep counter viewer with timeline scrubbing."""

import json
import os
from pathlib import Path
from flask import Flask, send_from_directory, jsonify, request, send_file
import subprocess
import sys

app = Flask(__name__, static_folder="web")

DATA_DIR = Path(os.environ.get("NOCAP_DATA", Path.home() / "nocap-data"))
RESULTS_DIR = DATA_DIR / "results"
VIDEOS_DIR = Path(os.environ.get("NOCAP_VIDEOS", Path.home() / "Downloads"))


@app.route("/")
def index():
    return send_from_directory("web", "index.html")


@app.route("/web/<path:filename>")
def static_files(filename):
    return send_from_directory("web", filename)


@app.route("/api/sessions")
def list_sessions():
    """List all analyzed sessions with their results."""
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
    """Get a single session's results."""
    path = RESULTS_DIR / f"{session_id}.json"
    if not path.exists():
        return jsonify({"error": "not found"}), 404
    with open(path) as f:
        data = json.load(f)
        data["id"] = session_id
    return jsonify(data)


@app.route("/api/video/<filename>")
def serve_video(filename):
    """Serve a video file with range request support."""
    # Check multiple locations
    for base in [VIDEOS_DIR, DATA_DIR / "videos", Path.home() / "Downloads"]:
        path = base / filename
        if path.exists():
            return send_file(path, conditional=True)
    return jsonify({"error": "video not found"}), 404


@app.route("/api/analyze", methods=["POST"])
def analyze_video():
    """Run rep counter on a video."""
    data = request.json
    video_path = data.get("video_path")
    exercise = data.get("exercise", "bench_press")

    if not video_path or not Path(video_path).exists():
        return jsonify({"error": "video not found"}), 400

    result = subprocess.run(
        [sys.executable, "count_reps.py", video_path, "-e", exercise, "--json"],
        capture_output=True, text=True, cwd=Path(__file__).parent,
    )

    if result.returncode != 0:
        return jsonify({"error": result.stderr}), 500

    # Parse JSON from output (it's after the text output)
    lines = result.stdout.strip().split("\n")
    json_start = next(i for i, l in enumerate(lines) if l.strip().startswith("{"))
    json_str = "\n".join(lines[json_start:])
    return jsonify(json.loads(json_str))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8765))
    print(f"nocap server running on http://localhost:{port}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Videos dir: {VIDEOS_DIR}")
    app.run(host="0.0.0.0", port=port, debug=True)
