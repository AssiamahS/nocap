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

# Full skeleton connections (like Jimna/Built red lines)
SKELETON_CONNECTIONS = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15),
    # Right arm
    (12, 14), (14, 16),
    # Left leg
    (23, 25), (25, 27),
    # Right leg
    (24, 26), (26, 28),
    # Hands
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    # Feet
    (27, 29), (27, 31),
    (28, 30), (28, 32),
]

# Key landmarks to draw as larger dots
KEY_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def draw_skeleton(frame, landmarks, w, h, left_angle, right_angle):
    """Draw full skeleton overlay on frame like Jimna/Built style."""
    # Draw connections (red lines)
    for a, b in SKELETON_CONNECTIONS:
        if a < len(landmarks) and b < len(landmarks):
            pt1 = (int(landmarks[a].x * w), int(landmarks[a].y * h))
            pt2 = (int(landmarks[b].x * w), int(landmarks[b].y * h))
            # Skip if landmarks are off-screen
            if (0 <= pt1[0] <= w and 0 <= pt1[1] <= h and
                    0 <= pt2[0] <= w and 0 <= pt2[1] <= h):
                cv2.line(frame, pt1, pt2, (0, 0, 255), 3)

    # Draw key joint dots (larger red circles with white border)
    for idx in KEY_LANDMARKS:
        if idx < len(landmarks):
            cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            if 0 <= cx <= w and 0 <= cy <= h:
                cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    # Draw angle arc at elbows
    for side, pts_idx in [("L", [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST]),
                          ("R", [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST])]:
        elbow = landmarks[pts_idx[1]]
        ex, ey = int(elbow.x * w), int(elbow.y * h)
        angle = left_angle if side == "L" else right_angle
        cv2.putText(frame, f"{angle:.0f}", (ex + 10, ey - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


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
        # Curl: extended (arm straight) ~130-170°, contracted (curled up) ~50-90°
        extended_threshold = min(max(p50 + (p90 - p50) * 0.3, 110), 150)
        contracted_threshold = min(max(p10 + (p50 - p10) * 0.3, 50), 100)
        min_transition_frames = 6
    elif exercise == "tricep_extension":
        # Tricep ext: extended (arms overhead straight) ~140-170°, contracted (bent behind head) ~40-80°
        extended_threshold = min(max(p50 + (p90 - p50) * 0.3, 100), 140)
        contracted_threshold = min(max(p10 + (p50 - p10) * 0.3, 40), 90)
        min_transition_frames = 8
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


def count_reps_wrist_tracking(landmarks_per_frame, frame_indices, fps):
    """Universal rep counter — tracks wrist position in space.

    Instead of tracking joint angles (which require knowing the exercise),
    this tracks where the wrist moves. Every exercise moves the weight
    in a repeating pattern. This finds that pattern automatically.

    Returns: (rep_count, rep_frame_indices, wrist_signal, axis_info)
    """
    if len(landmarks_per_frame) < 10:
        return 0, [], [], {}

    # Extract wrist positions for both sides, both axes
    signals = {}
    for side, idx in [("L", 15), ("R", 16)]:
        for axis, ai in [("x", 0), ("y", 1)]:
            key = f"{side}_{axis}"
            signals[key] = np.array([lm[idx][ai] for lm in landmarks_per_frame])

    # Find the axis with the most periodic motion
    # Score = number of clean peaks * prominence
    best_key = None
    best_score = 0
    best_peaks = []

    kernel = np.ones(5) / 5

    for key, sig in signals.items():
        smoothed = np.convolve(sig, kernel, mode='same')
        sig_range = np.max(smoothed) - np.min(smoothed)

        if sig_range < 0.03:  # Too little motion
            continue

        # Find peaks (high points) and valleys (low points)
        prominence = sig_range * 0.15
        peaks, props = find_peaks(smoothed, prominence=prominence, distance=8)
        valleys, _ = find_peaks(-smoothed, prominence=prominence, distance=8)

        # Score: prefer axes where peaks and valleys alternate cleanly
        # and there are a reasonable number of them
        n_cycles = min(len(peaks), len(valleys))
        if n_cycles == 0:
            continue

        avg_prominence = np.mean(props['prominences'][:n_cycles]) if len(props['prominences']) > 0 else 0
        score = n_cycles * avg_prominence

        if score > best_score:
            best_score = score
            best_key = key
            best_peaks = valleys  # valleys = bottom of each rep

    if best_key is None or best_score == 0:
        return 0, [], [], {}

    # Re-analyze the best axis with refined detection
    sig = signals[best_key]
    smoothed = np.convolve(sig, kernel, mode='same')
    sig_range = np.max(smoothed) - np.min(smoothed)

    # --- SET DETECTION: Only count reps during actual exercise sets ---
    # Use sliding window amplitude to find "active" periods.
    # Talking, walking, adjusting weights = low amplitude.
    # Actual reps = high amplitude rhythmic motion.
    window_size = int(fps * 2)  # 2-second sliding window
    if window_size < 10:
        window_size = 10

    amplitudes = np.zeros(len(smoothed))
    for i in range(len(smoothed)):
        start = max(0, i - window_size // 2)
        end = min(len(smoothed), i + window_size // 2)
        segment = smoothed[start:end]
        amplitudes[i] = np.max(segment) - np.min(segment)

    # Set detection: filter out talking/setup/walking noise.
    # Only count reps during periods of large rhythmic motion.
    peak_amplitude = np.percentile(amplitudes, 90)
    active_threshold = peak_amplitude * 0.35

    # Count what fraction of the video is "active"
    active_frames = np.sum(amplitudes >= active_threshold)
    active_fraction = active_frames / len(amplitudes)

    if active_fraction > 0.6:
        # Most of the video is exercise — don't filter (clean/trimmed clip)
        active_mask = np.ones(len(smoothed), dtype=bool)
    else:
        # Mixed video (talking + exercise) — only keep active regions
        active_mask = amplitudes >= active_threshold

    # Find peaks/valleys only in active regions
    prominence = sig_range * 0.15
    peaks, _ = find_peaks(smoothed, prominence=prominence, distance=8)
    valleys, _ = find_peaks(-smoothed, prominence=prominence, distance=8)

    # Filter: only keep peaks/valleys in active regions
    peaks = np.array([p for p in peaks if active_mask[p]])
    valleys = np.array([p for p in valleys if active_mask[p]])

    # --- CLUSTER FILTER ---
    # Real exercise reps come in dense bursts (sets). Noise peaks are scattered.
    # Find the largest cluster of valleys (bottom of reps) and keep only those.
    # Only apply on longer videos where noise is likely (>20s with >5 valleys).
    video_duration = len(smoothed) / max(fps, 1)
    if len(valleys) > 4 and video_duration > 20:
        # Cluster valleys: two valleys are in the same cluster if gap < 5 seconds
        max_gap = fps * 5
        clusters = []
        current_cluster = [valleys[0]]
        for i in range(1, len(valleys)):
            if valleys[i] - valleys[i-1] <= max_gap:
                current_cluster.append(valleys[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [valleys[i]]
        clusters.append(current_cluster)

        # Keep the largest cluster (that's the main exercise set)
        largest = max(clusters, key=len)
        if len(largest) >= 2 and len(largest) < len(valleys):
            valleys = np.array(largest)
            # Also filter peaks to the same time range
            start_frame = largest[0] - int(fps * 2)
            end_frame = largest[-1] + int(fps * 2)
            peaks = np.array([p for p in peaks if start_frame <= p <= end_frame])

    # Count reps as complete peak-valley pairs
    reps_pv = min(len(peaks), len(valleys))
    rep_indices = valleys[:reps_pv].tolist() if len(valleys) >= reps_pv else peaks[:reps_pv].tolist()

    # Normalize signal to 0-1 for the meter
    normalized = ((smoothed - np.min(smoothed)) / (sig_range + 1e-6)).tolist()

    axis_info = {
        "axis": best_key,
        "side": best_key.split("_")[0],
        "direction": best_key.split("_")[1],
        "range": round(float(sig_range), 4),
        "n_peaks": int(len(peaks)),
        "n_valleys": int(len(valleys)),
    }

    return reps_pv, rep_indices, normalized, axis_info


def process_video(video_path, exercise="bench_press", output_video=None, verbose=False, save_poses=None):
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
    all_landmarks = []  # Per-frame landmark arrays for wrist tracking
    pose_frames = []  # Per-frame 3D pose data for visualization
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
                all_landmarks.append([[lm.x, lm.y, lm.z] for lm in landmarks])

                # Save 3D pose data (sample every 3 frames to keep file size down)
                if save_poses and frame_idx % 3 == 0:
                    pose_frames.append({
                        "f": frame_idx,
                        "t": round(frame_idx / fps, 3),
                        "lm": [[round(lm.x, 4), round(lm.y, 4), round(lm.z, 4)]
                               for lm in landmarks],
                        "la": round(left_angle, 1),
                        "ra": round(right_angle, 1),
                    })

                if verbose and frame_idx % 30 == 0:
                    print(f"  Frame {frame_idx} ({frame_idx/fps:.1f}s): L={left_angle:.1f} R={right_angle:.1f}")

                # Draw skeleton on output video
                if writer:
                    draw_skeleton(frame, landmarks, width, height, left_angle, right_angle)

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

    rep_count_angle, peak_indices_angle, smoothed = count_reps_from_angles(primary_angles, exercise)

    # --- Wrist position tracking (universal, exercise-agnostic) ---
    rep_count_wrist, peak_indices_wrist, wrist_signal, axis_info = \
        count_reps_wrist_tracking(all_landmarks, frame_indices, fps)

    if axis_info:
        print(f"Wrist tracking: {axis_info['axis']} axis, {axis_info['n_peaks']} peaks, {axis_info['n_valleys']} valleys")
        print(f"  Angle-based: {rep_count_angle} reps | Wrist-based: {rep_count_wrist} reps")

    # Wrist tracking has set detection (filters out talking/setup noise).
    # Angle tracking does NOT have set detection, so it overcounts on videos
    # with non-exercise movement.
    # Default to wrist tracking when available — it's more universal and
    # has better noise filtering. Only fall back to angle if wrist detected 0.
    if rep_count_wrist > 0 and axis_info:
        rep_count = rep_count_wrist
        rep_indices_final = peak_indices_wrist
        method = "wrist_position"
        print(f"  -> Using WRIST tracking ({rep_count} reps)")
    else:
        rep_count = rep_count_angle
        rep_indices_final = peak_indices_angle
        method = "elbow_angle"
        print(f"  -> Using ANGLE tracking ({rep_count} reps, no wrist signal)")

    rep_timestamps = []
    for pi in rep_indices_final:
        if pi < len(frame_indices):
            rep_timestamps.append(round(frame_indices[pi] / fps, 2))

    result = {
        "video": str(video_path),
        "exercise": exercise,
        "reps": rep_count,
        "method": method,
        "reps_angle": rep_count_angle,
        "reps_wrist": rep_count_wrist,
        "rep_timestamps_sec": rep_timestamps,
        "duration_sec": round(duration, 2),
        "fps": round(fps, 1),
        "total_frames": total_frames,
        "pose_detection_rate": round(pose_detected_frames / max(total_frames, 1), 3),
        "primary_side": primary_side,
        "angle_range_deg": round(float(trimmed_range), 1),
        "exercise_start_sec": round(frame_indices[0] / fps, 2) if frame_indices else 0,
        "wrist_axis": axis_info.get("axis", ""),
    }

    # Save 3D pose data for visualization
    if save_poses and pose_frames:
        pose_data = {
            "video": str(video_path),
            "fps": round(fps, 1),
            "width": width,
            "height": height,
            "sample_rate": 3,
            "connections": SKELETON_CONNECTIONS,
            "frames": pose_frames,
            "wrist_signal": wrist_signal if wrist_signal else [],
            "wrist_axis": axis_info.get("axis", ""),
            "method": method,
        }
        with open(save_poses, 'w') as f:
            json.dump(pose_data, f)
        print(f"Saved {len(pose_frames)} pose frames to {save_poses}")

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
                        choices=["bench_press", "push_up", "forearm_curl", "tricep_extension"],
                        default="bench_press")
    parser.add_argument("-o", "--output", help="Save annotated video")
    parser.add_argument("-p", "--save-poses", help="Save 3D pose data to JSON file")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    result = process_video(video_path, args.exercise, args.output, args.verbose, args.save_poses)

    if args.json:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
