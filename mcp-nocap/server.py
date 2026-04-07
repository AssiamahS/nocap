#!/usr/bin/env python3
"""nocap MCP server — lets Claude see rep counting results, pose data, and logs."""

import json
import sys
import os
from pathlib import Path

# MCP protocol via stdin/stdout
def read_message():
    header = ""
    while True:
        line = sys.stdin.readline()
        if not line:
            return None
        header += line
        if line.strip() == "":
            break
    content_length = 0
    for line in header.strip().split("\n"):
        if line.lower().startswith("content-length:"):
            content_length = int(line.split(":")[1].strip())
    if content_length == 0:
        return None
    body = sys.stdin.read(content_length)
    return json.loads(body)

def write_message(msg):
    body = json.dumps(msg)
    sys.stdout.write(f"Content-Length: {len(body)}\r\n\r\n{body}")
    sys.stdout.flush()

DATA_DIR = Path(os.environ.get("NOCAP_DATA", Path.home() / "nocap-data"))
RESULTS_DIR = DATA_DIR / "results"
NOCAP_DIR = Path(__file__).parent.parent
WEB_DIR = NOCAP_DIR / "web"


def list_sessions():
    sessions = []
    if RESULTS_DIR.exists():
        for f in sorted(RESULTS_DIR.glob("*.json")):
            with open(f) as fh:
                data = json.load(fh)
                data["id"] = f.stem
                sessions.append(data)
    return sessions


def get_session(session_id):
    path = RESULTS_DIR / f"{session_id}.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
            data["id"] = session_id
            return data
    return {"error": "not found"}


def get_pose_summary(session_id):
    """Get a summary of pose data for a session — angles and wrist positions over time."""
    # Try multiple paths
    for pattern in [f"{session_id}_poses.json", f"{session_id.replace('_bench_press','')}_poses.json"]:
        path = WEB_DIR / pattern
        if path.exists():
            with open(path) as f:
                data = json.load(f)

            frames = data.get("frames", [])
            wrist_signal = data.get("wrist_signal", [])

            # Sample every ~2 seconds for a readable summary
            fps = data.get("fps", 30)
            sample_rate = data.get("sample_rate", 3)
            summary_frames = []
            for i in range(0, len(frames), max(1, int(fps / sample_rate * 2))):
                f = frames[i]
                summary_frames.append({
                    "time": f["t"],
                    "left_elbow_angle": f["la"],
                    "right_elbow_angle": f["ra"],
                    "left_wrist": f["lm"][15][:2] if len(f["lm"]) > 15 else None,
                    "right_wrist": f["lm"][16][:2] if len(f["lm"]) > 16 else None,
                })

            return {
                "session_id": session_id,
                "total_frames": len(frames),
                "wrist_axis": data.get("wrist_axis", ""),
                "method": data.get("method", ""),
                "wrist_signal_length": len(wrist_signal),
                "sampled_frames": summary_frames,
            }

    # Search
    for f in WEB_DIR.glob("*poses*"):
        if session_id in f.stem:
            return get_pose_summary_from_file(f, session_id)

    return {"error": "pose data not found"}


def get_pose_summary_from_file(path, session_id):
    with open(path) as f:
        data = json.load(f)
    return {
        "session_id": session_id,
        "total_frames": len(data.get("frames", [])),
        "wrist_axis": data.get("wrist_axis", ""),
        "method": data.get("method", ""),
    }


def handle_request(msg):
    method = msg.get("method", "")
    params = msg.get("params", {})
    msg_id = msg.get("id")

    if method == "initialize":
        write_message({
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "nocap", "version": "1.8"},
            }
        })
    elif method == "notifications/initialized":
        pass  # No response needed
    elif method == "tools/list":
        write_message({
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "tools": [
                    {
                        "name": "list_sessions",
                        "description": "List all nocap workout sessions with rep counts, timestamps, and exercise types",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                    {
                        "name": "get_session",
                        "description": "Get detailed results for a specific session including rep timestamps and detection method",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"session_id": {"type": "string", "description": "Session ID"}},
                            "required": ["session_id"],
                        },
                    },
                    {
                        "name": "get_pose_data",
                        "description": "Get sampled pose data for a session — elbow angles and wrist positions over time. Use this to see what the rep counter sees.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"session_id": {"type": "string", "description": "Session ID"}},
                            "required": ["session_id"],
                        },
                    },
                    {
                        "name": "count_reps",
                        "description": "Run the rep counter on a video file. Returns rep count, timestamps, and detection method.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "video_path": {"type": "string", "description": "Absolute path to video file"},
                                "exercise": {"type": "string", "enum": ["bench_press", "push_up", "forearm_curl", "tricep_extension"], "default": "bench_press"},
                            },
                            "required": ["video_path"],
                        },
                    },
                ]
            }
        })
    elif method == "tools/call":
        tool_name = params.get("name", "")
        args = params.get("arguments", {})

        if tool_name == "list_sessions":
            result = list_sessions()
        elif tool_name == "get_session":
            result = get_session(args.get("session_id", ""))
        elif tool_name == "get_pose_data":
            result = get_pose_summary(args.get("session_id", ""))
        elif tool_name == "count_reps":
            import subprocess
            video_path = args.get("video_path", "")
            exercise = args.get("exercise", "bench_press")
            proc = subprocess.run(
                [sys.executable, str(NOCAP_DIR / "count_reps.py"), video_path, "-e", exercise, "--json"],
                capture_output=True, text=True, timeout=600,
                env={**os.environ, "PATH": os.environ.get("PATH", "")},
            )
            if proc.returncode == 0:
                lines = proc.stdout.strip().split("\n")
                json_start = next((i for i, l in enumerate(lines) if l.strip().startswith("{")), None)
                if json_start is not None:
                    result = json.loads("\n".join(lines[json_start:]))
                else:
                    result = {"output": proc.stdout[-500:]}
            else:
                result = {"error": proc.stderr[-500:]}
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        write_message({
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "content": [{"type": "text", "text": json.dumps(result, indent=2)}]
            }
        })
    else:
        if msg_id:
            write_message({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {}
            })


def main():
    while True:
        msg = read_message()
        if msg is None:
            break
        handle_request(msg)


if __name__ == "__main__":
    main()
