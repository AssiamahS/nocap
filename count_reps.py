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

    A rep = extended (high angle) -> contracted (low angle) -> extended (high angle).
    Handles edge cases: video starts mid-rep, last rep cut off by video end.
    Only counts reps where both the contraction and extension are fully visible.
    """
    if len(angles) < 10:
        return 0, [], []

    kernel = np.ones(7) / 7
    smoothed = np.convolve(angles, kernel, mode='same')

    angle_range = np.max(smoothed) - np.min(smoothed)
    if angle_range < 15:
        return 0, [], []

    # Dynamic thresholds from signal percentiles, with exercise-specific bounds.
    p10 = np.percentile(smoothed, 10)
    p50 = np.percentile(smoothed, 50)
    p90 = np.percentile(smoothed, 90)

    if exercise == "bench_press":
        # Bench: extended ~100-160°, contracted ~40-90°
        extended_threshold = min(max(p50 + (p90 - p50) * 0.3, 90), 130)
        contracted_threshold = min(max(p10 + (p50 - p10) * 0.3, 40), 80)
        min_transition_frames = 10
    elif exercise == "forearm_curl":
        # Curl: extended ~130-170°, contracted ~50-90°
        extended_threshold = min(max(p50 + (p90 - p50) * 0.3, 110), 150)
        contracted_threshold = min(max(p10 + (p50 - p10) * 0.3, 50), 100)
        min_transition_frames = 6
    else:  # push_up
        extended_threshold = min(max(p50 + (p90 - p50) * 0.3, 140), 165)
        contracted_threshold = min(max(p10 + (p50 - p10) * 0.3, 80), 120)
        min_transition_frames = 10

    # State machine
    STATE_SEEKING_EXTENDED = 0   # Looking for first full extension
    STATE_EXTENDED = 1           # At top/extended, waiting to contract
    STATE_CONTRACTED = 2         # At bottom/contracted, waiting to extend

    state = STATE_SEEKING_EXTENDED
    reps = 0
    rep_indices = []
    last_extended_idx = 0
    contracted_idx = 0

    for i, angle in enumerate(smoothed):
        if state == STATE_SEEKING_EXTENDED:
            if angle >= extended_threshold:
                state = STATE_EXTENDED
                last_extended_idx = i

        elif state == STATE_EXTENDED:
            # Two ways to detect contraction:
            # 1. Absolute: angle drops below contracted_threshold
            # 2. Relative: angle drops by at least 30% of the range from last extension
            #    (handles grinder reps that don't go as deep)
            last_ext_angle = smoothed[last_extended_idx]
            relative_drop = last_ext_angle - angle
            min_relative_drop = (extended_threshold - contracted_threshold) * 0.4

            is_contracted = (angle <= contracted_threshold) or (relative_drop >= min_relative_drop)

            if is_contracted and (i - last_extended_idx) >= min_transition_frames:
                state = STATE_CONTRACTED
                contracted_idx = i

        elif state == STATE_CONTRACTED:
            if angle >= extended_threshold and (i - contracted_idx) >= min_transition_frames:
                # Full rep completed: extended -> contracted -> extended
                # Enforce minimum total rep time (extension -> contraction -> extension)
                total_rep_frames = i - last_extended_idx
                min_total_rep_frames = min_transition_frames * 3  # ~1 second minimum
                if total_rep_frames >= min_total_rep_frames:
                    reps += 1
                    rep_indices.append(contracted_idx)
                state = STATE_EXTENDED
                last_extended_idx = i

    # Handle incomplete final rep (e.g., racking the bar or video ends mid-recovery).
    # Only count if we saw significant recovery (>= 50% of the way back to extension).
    # Do NOT count if video just cuts off at the bottom — that rep isn't confirmed.
    if state == STATE_CONTRACTED:
        remaining = smoothed[contracted_idx:]
        if len(remaining) > 5:
            max_after = np.max(remaining)
            recovery = max_after - smoothed[contracted_idx]
            needed = extended_threshold - smoothed[contracted_idx]
            if needed > 0 and recovery / needed >= 0.5:
                reps += 1
                rep_indices.append(contracted_idx)

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
