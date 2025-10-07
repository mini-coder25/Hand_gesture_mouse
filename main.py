import argparse
import math
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from typing import cast


# Indices for Mediapipe hand landmarks
TIP_THUMB = 4
TIP_INDEX = 8
TIP_MIDDLE = 12
BASE_INDEX = 5  # index MCP
BASE_PINKY = 17  # pinky MCP
WRIST = 0


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def smooth(prev: float, curr: float, alpha: float) -> float:
    return lerp(prev, curr, alpha)


def distance(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def palm_width(landmarks_px) -> float:
    # Approximate palm width as distance between index MCP and pinky MCP
    return distance(landmarks_px[BASE_INDEX], landmarks_px[BASE_PINKY])


def count_fingers_up(landmarks_norm) -> int:
    # Simple heuristic: compare each fingertip y to its PIP y (for non-thumb)
    # Using normalized coords (y up = smaller value if image not flipped)
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    up = 0
    for tip, pip in zip(tips, pips):
        if landmarks_norm[tip][1] < landmarks_norm[pip][1]:
            up += 1
    # Thumb: check if tip is away from palm horizontally (x distance from index MCP)
    thumb_tip = landmarks_norm[4]
    index_mcp = landmarks_norm[5]
    if abs(thumb_tip[0] - index_mcp[0]) > 0.1:
        up += 1
    return up


def main():
    parser = argparse.ArgumentParser(description="Gesture-controlled mouse with MediaPipe Hands")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--no-flip", action="store_true", help="Disable horizontal flip of preview")
    parser.add_argument("--alpha", type=float, default=0.6, help="Smoothing factor 0..1 (higher = faster)")
    parser.add_argument("--show", type=int, default=1, help="Show preview window (1) or not (0)")
    parser.add_argument("--left-scale", type=float, default=0.33, help="Left click threshold as fraction of palm width")
    parser.add_argument("--right-scale", type=float, default=0.33, help="Right click threshold as fraction of palm width")
    parser.add_argument("--drag-hold", type=int, default=0, help="Enable drag-and-hold for index pinch (0/1)")
    parser.add_argument("--width", type=int, default=1280, help="Camera capture width")
    parser.add_argument("--height", type=int, default=720, help="Camera capture height")
    parser.add_argument("--fps", type=int, default=60, help="Requested camera FPS (if supported)")
    parser.add_argument("--speed-gain", type=float, default=1.8, help="Cursor speed gain multiplier")
    # Scrolling options
    parser.add_argument("--scroll", type=int, default=1, help="Enable fist-to-scroll gesture (0/1)")
    parser.add_argument("--scroll-sens", type=float, default=0.7, help="Scroll sensitivity (higher=faster)")
    parser.add_argument("--scroll-invert", action="store_true", help="Invert scroll direction")
    args = parser.parse_args()

    pyautogui.FAILSAFE = False
    screen_w, screen_h = pyautogui.size()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    # Try to boost frame rate and throughput
    try:
        fourcc_func = getattr(cv2, "VideoWriter_fourcc", None)
        if callable(fourcc_func):
            fourcc_obj = fourcc_func(*"MJPG")  # runtime int; stubs may be loose
            fourcc_val = float(cast(int, fourcc_obj))
            cap.set(cv2.CAP_PROP_FOURCC, fourcc_val)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    except Exception:
        pass
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try a different index with --camera.")

    # MediaPipe setup - use runtime attribute access to work with different versions
    mp_hands = getattr(mp.solutions, 'hands', None) or getattr(mp.python.solutions, 'hands')
    mp_draw = getattr(mp.solutions, 'drawing_utils', None) or getattr(mp.python.solutions, 'drawing_utils')
    mp_styles = getattr(mp.solutions, 'drawing_styles', None) or getattr(mp.python.solutions, 'drawing_styles')

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        model_complexity=1,
    )

    prev_x, prev_y = None, None
    last_left_click = 0.0
    last_right_click = 0.0
    click_cooldown = 0.25  # seconds
    dragging = False
    control_enabled = False
    toggle_hold_frames = 0
    toggle_hold_target = 10

    # New: scroll state
    scrolling = False
    last_scroll_y = None
    scroll_residual = 0.0

    fps_deque = deque(maxlen=30)
    last_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if not args.no_flip:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            h, w = frame.shape[:2]
            now = time.time()
            dt = now - last_time
            last_time = now
            fps_deque.append(1.0 / dt if dt > 0 else 0.0)

            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0]
                # Normalized 0..1
                landmarks_norm = [(p.x, p.y) for p in lm.landmark]
                # Convert to pixel coords
                landmarks_px = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]

                if args.show:
                    mp_draw.draw_landmarks(
                        frame,
                        result.multi_hand_landmarks[0],
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                # Toggle gesture: open palm (>=4 fingers up) held briefly
                fingers = count_fingers_up(landmarks_norm)
                if fingers >= 4:
                    toggle_hold_frames += 1
                else:
                    if toggle_hold_frames >= toggle_hold_target:
                        control_enabled = not control_enabled
                        dragging = False
                    toggle_hold_frames = 0

                # Cursor movement using index tip
                idx_x_px, idx_y_px = landmarks_px[TIP_INDEX]
                # Map from camera space to screen space
                screen_x = clamp(int(idx_x_px / w * screen_w), 0, screen_w - 1)
                screen_y = clamp(int(idx_y_px / h * screen_h), 0, screen_h - 1)

                if prev_x is None or prev_y is None:
                    prev_x, prev_y = screen_x, screen_y
                else:
                    k = clamp(args.alpha * args.speed_gain, 0.01, 1.0)
                    prev_x = smooth(prev_x, screen_x, k)
                    prev_y = smooth(prev_y, screen_y, k)

                # New: fist-to-scroll gesture (0–1 fingers up)
                if control_enabled and args.scroll and fingers <= 1:
                    curr_y = landmarks_px[WRIST][1]  # use wrist vertical movement
                    if last_scroll_y is not None:
                        dy = last_scroll_y - curr_y  # up -> positive
                        # Accumulate fractional steps for smooth scrolling
                        scroll_residual += dy * args.scroll_sens
                        steps = int(scroll_residual)
                        if steps != 0:
                            direction = 1 if not args.scroll_invert else -1
                            pyautogui.scroll(steps * direction)
                            scroll_residual -= steps
                    last_scroll_y = curr_y
                    # Prevent accidental drags while scrolling
                    if dragging:
                        pyautogui.mouseUp()
                        dragging = False
                    scrolling = True
                else:
                    scrolling = False
                    last_scroll_y = None
                    scroll_residual = 0.0

                if control_enabled and not scrolling:
                    pyautogui.moveTo(prev_x, prev_y)

                # Click gestures use palm-relative thresholds
                pw = max(palm_width(landmarks_px), 1.0)
                left_thresh = pw * args.left_scale
                right_thresh = pw * args.right_scale

                thumb_px = landmarks_px[TIP_THUMB]
                index_px = landmarks_px[TIP_INDEX]
                middle_px = landmarks_px[TIP_MIDDLE]
                d_thumb_index = distance(thumb_px, index_px)
                d_thumb_middle = distance(thumb_px, middle_px)

                if control_enabled:
                    # Suppress clicks while scrolling
                    if not scrolling:
                        # Left click or drag
                        if d_thumb_index < left_thresh:
                            if args.drag_hold:
                                if not dragging:
                                    pyautogui.mouseDown()
                                    dragging = True
                            else:
                                if (now - last_left_click) > click_cooldown:
                                    pyautogui.click()
                                    last_left_click = now
                        else:
                            if dragging:
                                pyautogui.mouseUp()
                                dragging = False

                        # Right click
                        if d_thumb_middle < right_thresh and (now - last_right_click) > click_cooldown:
                            pyautogui.click(button="right")
                            last_right_click = now

                # Overlays
                if args.show:
                    status = f"Ctrl: {'ON' if control_enabled else 'OFF'}  Drag: {'ON' if dragging else 'OFF'}  Scroll: {'ON' if scrolling else 'OFF'}"
                    fps = sum(fps_deque) / max(1, len(fps_deque))
                    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if control_enabled else (0, 0, 255), 2)
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.circle(frame, (int(prev_x / screen_w * w), int(prev_y / screen_h * h)), 6, (0, 255, 255), 2)

            else:
                toggle_hold_frames = 0
                scrolling = False
                last_scroll_y = None
                scroll_residual = 0.0

            if args.show:
                cv2.imshow("Gesture Cam", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    break

    finally:
        if dragging:
            pyautogui.mouseUp()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
