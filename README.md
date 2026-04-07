# nocap

AI-powered exercise rep counter. No cap on your gains.

## V1 - Video Rep Counter

Processes uploaded workout videos and counts reps using MediaPipe Pose detection.

### Supported Exercises
- Bench press (Smith machine, flat/incline)
- Forearm curls (EZ bar)
- Push-ups (coming soon)

### How It Works
1. MediaPipe Pose detects 33 body landmarks per frame
2. Tracks elbow angle (shoulder → elbow → wrist)
3. State machine detects full rep cycles: lockout → bottom → lockout
4. Auto-detects which side of body is visible to camera
5. Skips setup time before first rep

### Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Usage

```bash
# Count bench press reps
python count_reps.py video.mov -e bench_press

# Count forearm curls
python count_reps.py video.mov -e forearm_curl

# Generate annotated output video with landmarks
python count_reps.py video.mov -e bench_press -o output.mp4

# JSON output
python count_reps.py video.mov -e bench_press --json

# Verbose (show angles every second)
python count_reps.py video.mov -e bench_press -v
```

### Tech Stack
- **MediaPipe Pose** - Google's free pose estimation (runs 100% locally, no cloud)
- **OpenCV** - Video processing
- **scipy** - Signal processing for rep detection

### Roadmap
- V1: Video upload rep counting (current)
- V2: Live iOS app with Apple Vision framework
- V3: Leaderboards, weight tracking, manual input
