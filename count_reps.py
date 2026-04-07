#!/usr/bin/env python3
"""
nocap v1 - Exercise Rep Counter
Uses MediaPipe Pose Landmarker to detect body landmarks and count reps from video.
Supports: bench press, push-ups, forearm curls
"""

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import find_peaks
import argparse
import json
import sys
import os
import urllib.request
from pathlib import Path

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
MODEL_PATH = Path(__file__).parent / "pose_landmarker.task"

# Landmark indices
LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15
RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16


def ensure_model():
    if not MODEL_PATH.exists():
        print("Downloading pose model (~30MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def calculate_angle(a, b, c):
    """Calculate angle at point b given three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def detect_exercise_start(angles, window=15, threshold=5.0):
    """Detect when actual exercise movement begins (skip setup time)."""
    for i in range(window, len(angles)):
        if np.std(angles[i - window:i]) > threshold:
            return max(0, i - window)
    return 0


def count_reps_from_angles(angles, exercise="bench_press"):
    """Count reps using a state machine that requires full rep cycles.

    A rep = lockout (high angle) -> bottom (low angle) -> lockout (high angle).
    This prevents counting setup/unrack, partial movements, and post-rack noise.
    """
    if len(angles) < 10:
        return 0, [], []

    kernel = np.ones(7) / 7
    smoothed = np.convolve(angles, kernel, mode='same')

    angle_range = np.max(smoothed) - np.min(smoothed)
    if angle_range < 15:
        return 0, [], []

    # Exercise-specific thresholds
    if exercise == "bench_press":
        lockout_threshold = 100   # Above this = locked out
        bottom_threshold = 70     # Below this = bottom of rep
        min_rep_frames = 10       # Minimum frames for a rep
    elif exercise == "forearm_curl":
        lockout_threshold = 120
        bottom_threshold = 80
        min_rep_frames = 8
    else:  # push_up
        lockout_threshold = 150
        bottom_threshold = 100
        min_rep_frames = 12

    # Dynamic threshold: use percentiles of the actual signal
    p25 = np.percentile(smoothed, 25)
    p75 = np.percentile(smoothed, 75)
    midpoint = (p25 + p75) / 2

    # Adjust thresholds based on actual data
    lockout_threshold = max(lockout_threshold, midpoint + (p75 - p25) * 0.3)
    bottom_threshold = min(bottom_threshold, midpoint - (p75 - p25) * 0.1)

    # State machine
    STATE_WAITING_LOCKOUT = 0  # Waiting for first lockout
    STATE_LOCKED_OUT = 1       # At top, waiting to go down
    STATE_AT_BOTTOM = 2        # At bottom, waiting to come back up

    state = STATE_WAITING_LOCKOUT
    reps = 0
    rep_indices = []
    last_lockout_idx = 0
    bottom_idx = 0

    for i, angle in enumerate(smoothed):
        if state == STATE_WAITING_LOCKOUT:
            if angle >= lockout_threshold:
                state = STATE_LOCKED_OUT
                last_lockout_idx = i

        elif state == STATE_LOCKED_OUT:
            if angle <= bottom_threshold and (i - last_lockout_idx) >= min_rep_frames:
                state = STATE_AT_BOTTOM
                bottom_idx = i

        elif state == STATE_AT_BOTTOM:
            if angle >= lockout_threshold and (i - bottom_idx) >= min_rep_frames:
                # Full rep completed: lockout -> bottom -> lockout
                reps += 1
                rep_indices.append(bottom_idx)
                state = STATE_LOCKED_OUT
                last_lockout_idx = i

    # Check if there's an incomplete rep at the end (like racking counts as lockout)
    # If we're at bottom and the angle goes up significantly but maybe not to full lockout
    if state == STATE_AT_BOTTOM:
        remaining = smoothed[bottom_idx:]
        if len(remaining) > 5:
            max_after_bottom = np.max(remaining)
            # If it came back up at least 60% of the way to lockout, count it
            recovery = max_after_bottom - smoothed[bottom_idx]
            needed = lockout_threshold - smoothed[bottom_idx]
            if needed > 0 and recovery / needed >= 0.5:
                reps += 1
                rep_indices.append(bottom_idx)

    return reps, rep_indices, smoothed.tolist()


def process_video(video_path, exercise="bench_press", output_video=None, verbose=False):
    """Process a video file and count exercise reps."""
    ensure_model()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video: {Path(video_path).name}")
    print(f"Resolution: {width}x{height}, FPS: {fps:.1f}, Duration: {duration:.1f}s, Frames: {total_frames}")
    print(f"Exercise: {exercise}")
    print(f"Processing...")

    # Set up PoseLandmarker with new tasks API
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    left_angles = []
    right_angles = []
    frame_indices = []
    pose_detected_frames = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to MediaPipe Image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Detect pose
            timestamp_ms = int(frame_idx * 1000 / fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                pose_detected_frames += 1
                landmarks = result.pose_landmarks[0]

                # Calculate elbow angles
                left_pts = [
                    [landmarks[LEFT_SHOULDER].x, landmarks[LEFT_SHOULDER].y],
                    [landmarks[LEFT_ELBOW].x, landmarks[LEFT_ELBOW].y],
                    [landmarks[LEFT_WRIST].x, landmarks[LEFT_WRIST].y],
                ]
                right_pts = [
                    [landmarks[RIGHT_SHOULDER].x, landmarks[RIGHT_SHOULDER].y],
                    [landmarks[RIGHT_ELBOW].x, landmarks[RIGHT_ELBOW].y],
                    [landmarks[RIGHT_WRIST].x, landmarks[RIGHT_WRIST].y],
                ]

                left_angle = calculate_angle(*left_pts)
                right_angle = calculate_angle(*right_pts)

                left_angles.append(left_angle)
                right_angles.append(right_angle)
                frame_indices.append(frame_idx)

                if verbose and frame_idx % 30 == 0:
                    print(f"  Frame {frame_idx} ({frame_idx/fps:.1f}s): L={left_angle:.1f} R={right_angle:.1f}")

                # Draw on output video
                if writer:
                    for lm in landmarks:
                        cx, cy = int(lm.x * width), int(lm.y * height)
                        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
                    cv2.putText(frame, f"L: {left_angle:.0f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"R: {right_angle:.0f}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if writer:
                writer.write(frame)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  {frame_idx}/{total_frames} ({100*frame_idx/total_frames:.0f}%)", end="\r")

    cap.release()
    if writer:
        writer.release()

    print(f"\nPose detected in {pose_detected_frames}/{total_frames} frames ({100*pose_detected_frames/max(total_frames,1):.0f}%)")

    if not left_angles:
        print("No pose landmarks detected.")
        return {"reps": 0, "exercise": exercise, "error": "no_pose_detected"}

    angles_l = np.array(left_angles)
    angles_r = np.array(right_angles)

    # Pick the side closer to camera (better landmark tracking).
    # The occluded side tends to have unrealistically low angles (landmarks collapse).
    # A healthy bench press / curl should have median elbow angle ~60-100°.
    # Use the side with the higher median (more physically realistic).
    left_median = np.median(angles_l)
    right_median = np.median(angles_r)

    if right_median > left_median:
        primary_angles, primary_side = angles_r, "right"
    else:
        primary_angles, primary_side = angles_l, "left"

    trimmed_range = np.percentile(primary_angles, 95) - np.percentile(primary_angles, 5)
    print(f"Using {primary_side} side (median: {np.median(primary_angles):.1f}°, range: {trimmed_range:.1f}°)")

    # Detect exercise start
    start_idx = detect_exercise_start(primary_angles)
    if start_idx > 0:
        start_time = frame_indices[start_idx] / fps
        print(f"Exercise starts at {start_time:.1f}s (skipping {start_time:.1f}s of setup)")
        primary_angles = primary_angles[start_idx:]
        frame_indices = frame_indices[start_idx:]

    rep_count, peak_indices, smoothed = count_reps_from_angles(primary_angles, exercise)

    rep_timestamps = []
    for pi in peak_indices:
        if pi < len(frame_indices):
            rep_timestamps.append(round(frame_indices[pi] / fps, 2))

    result = {
        "video": str(video_path),
        "exercise": exercise,
        "reps": rep_count,
        "rep_timestamps_sec": rep_timestamps,
        "duration_sec": round(duration, 2),
        "fps": round(fps, 1),
        "total_frames": total_frames,
        "pose_detection_rate": round(pose_detected_frames / max(total_frames, 1), 3),
        "primary_side": primary_side,
        "angle_range_deg": round(float(trimmed_range), 1),
        "exercise_start_sec": round(frame_indices[0] / fps, 2) if frame_indices else 0,
    }

    print(f"\n{'='*40}")
    print(f"  REPS COUNTED: {rep_count}")
    if rep_timestamps:
        print(f"  Rep times: {', '.join(f'{t:.1f}s' for t in rep_timestamps)}")
    print(f"{'='*40}")

    return result


def main():
    parser = argparse.ArgumentParser(description="nocap v1 - Exercise Rep Counter")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("-e", "--exercise",
                        choices=["bench_press", "push_up", "forearm_curl"],
                        default="bench_press")
    parser.add_argument("-o", "--output", help="Save annotated video")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    result = process_video(video_path, args.exercise, args.output, args.verbose)

    if args.json:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
