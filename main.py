from __future__ import annotations

import math
import os
import time
import urllib.request
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

try:
    import pyautogui

    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0
except Exception:
    pyautogui = None


MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
MODEL_PATH = Path("assets") / "hand_landmarker.task"

THUMB_TIP = 4
THUMB_IP = 3
THUMB_MCP = 2
INDEX_TIP = 8
INDEX_PIP = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10
RING_TIP = 16
RING_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18
WRIST = 0


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def ensure_model(model_path: Path) -> None:
    if model_path.exists() and model_path.stat().st_size > 0:
        return

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand model to {model_path} ...")
    urllib.request.urlretrieve(MODEL_URL, str(model_path))


@dataclass
class TrackedHand:
    landmarks: np.ndarray
    label: str


@dataclass
class HandState:
    label: str
    landmarks: np.ndarray
    thumb_only: bool
    thumb_direction: Optional[str]
    index_extended: bool
    middle_extended: bool
    ring_extended: bool
    pinky_extended: bool
    open_palm: bool
    index_only: bool
    v_sign: bool
    fist_closed: bool

    @property
    def wrist(self) -> np.ndarray:
        return self.landmarks[WRIST, :2]


class HandTracker:
    """Tracks hands via mp.solutions when available, with tasks fallback."""

    def __init__(self) -> None:
        self.mode = "tasks"
        self.solution_hands = None
        self.task_detector = None
        self.timestamp_ms = int(time.time() * 1000)

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.mode = "solutions"
            self.solution_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
            )
            return

        ensure_model(MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.task_detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> List[TrackedHand]:
        if self.mode == "solutions":
            return self._detect_with_solutions(frame_bgr)
        return self._detect_with_tasks(frame_bgr)

    def _detect_with_solutions(self, frame_bgr: np.ndarray) -> List[TrackedHand]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.solution_hands.process(rgb)
        hands: List[TrackedHand] = []

        if not result.multi_hand_landmarks:
            return hands

        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            label = "Unknown"
            if result.multi_handedness and idx < len(result.multi_handedness):
                classification = result.multi_handedness[idx].classification
                if classification:
                    label = classification[0].label

            points = np.array(
                [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark],
                dtype=np.float32,
            )
            hands.append(TrackedHand(landmarks=points, label=label))

        return hands

    def _detect_with_tasks(self, frame_bgr: np.ndarray) -> List[TrackedHand]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self.timestamp_ms += 33
        result = self.task_detector.detect_for_video(
            mp_image, self.timestamp_ms)

        hands: List[TrackedHand] = []
        for idx, landmarks in enumerate(result.hand_landmarks):
            label = "Unknown"
            if idx < len(result.handedness) and result.handedness[idx]:
                label = result.handedness[idx][0].category_name

            points = np.array([(lm.x, lm.y, lm.z)
                              for lm in landmarks], dtype=np.float32)
            hands.append(TrackedHand(landmarks=points, label=label))

        return hands

    def close(self) -> None:
        if self.solution_hands is not None:
            self.solution_hands.close()
        if self.task_detector is not None:
            self.task_detector.close()


class HydraStateMachine:
    def __init__(self) -> None:
        self.last_fire: Dict[str, float] = {}
        self.prev_active: Set[str] = set()

        self.priority = [
            "TWO_INDEX_PLAY",
            "FIST_PAUSE",
            "V_SIGN_SPOTIFY",
            "THUMB_RIGHT_NEXT",
            "THUMB_LEFT_PREV",
            "THUMB_UP_VOL",
            "THUMB_DOWN_VOL",
        ]

        self.meta = {
            "TWO_INDEX_PLAY": {"cmd": "PLAY", "repeat": False, "cooldown": 0.7},
            "FIST_PAUSE": {"cmd": "PAUSE", "repeat": False, "cooldown": 0.7},
            "V_SIGN_SPOTIFY": {"cmd": "SPOTIFY", "repeat": False, "cooldown": 1.5},
            "THUMB_RIGHT_NEXT": {"cmd": "NEXT", "repeat": False, "cooldown": 0.7},
            "THUMB_LEFT_PREV": {"cmd": "PREV", "repeat": False, "cooldown": 0.7},
            "THUMB_UP_VOL": {"cmd": "VOL_UP", "repeat": True, "cooldown": 0.15},
            "THUMB_DOWN_VOL": {"cmd": "VOL_DOWN", "repeat": True, "cooldown": 0.15},
        }

    def resolve(self, active: Set[str], now: float) -> Optional[str]:
        command: Optional[str] = None

        for gesture in self.priority:
            if gesture not in active:
                continue

            item = self.meta[gesture]
            cmd = str(item["cmd"])
            repeatable = bool(item["repeat"])
            cooldown = float(item["cooldown"])
            was_active = gesture in self.prev_active
            elapsed = now - self.last_fire.get(cmd, 0.0)

            if (repeatable or not was_active) and elapsed >= cooldown:
                self.last_fire[cmd] = now
                command = cmd
            break

        self.prev_active = set(active)
        return command


class CommandExecutor:
    def __init__(self) -> None:
        self.last_text = ""
        self.last_text_time = 0.0

    def execute(self, command: str, now: float) -> bool:
        try:
            if command == "SPOTIFY":
                if os.name == "nt":
                    os.startfile("spotify:")
                else:
                    webbrowser.open("spotify:")
            elif pyautogui is not None:
                if command == "VOL_UP":
                    pyautogui.press("volumeup")
                elif command == "VOL_DOWN":
                    pyautogui.press("volumedown")
                elif command == "NEXT":
                    pyautogui.press("nexttrack")
                elif command == "PREV":
                    pyautogui.press("prevtrack")
                elif command in ("PLAY", "PAUSE"):
                    pyautogui.press("playpause")
                else:
                    return False
            else:
                return False

            self.last_text = command
            self.last_text_time = now
            return True
        except Exception:
            return False

    def recent_text(self, now: float) -> str:
        if (now - self.last_text_time) <= 0.8 and self.last_text:
            return self.last_text
        return ""


class LiquidVisualizer:
    """Draws transparent layered waves with perlin-like oscillation."""

    def __init__(self) -> None:
        self.phase = 0.0

    def _noise(self, t: float, layer: int) -> float:
        x = (t * 7.0) + (layer * 0.9)
        n = math.sin(x + (self.phase * 1.2))
        n += 0.55 * math.sin((x * 2.2) - (self.phase * 0.8))
        n += 0.28 * math.cos((x * 4.4) + (self.phase * 1.5))
        return n / 1.83

    def draw_water(self, target: np.ndarray, p1_norm: np.ndarray, p2_norm: np.ndarray) -> None:
        self.phase += 0.14

        h, w = target.shape[:2]
        p1 = np.array([p1_norm[0] * w, p1_norm[1] * h], dtype=np.float32)
        p2 = np.array([p2_norm[0] * w, p2_norm[1] * h], dtype=np.float32)

        vec = p2 - p1
        distance = float(np.linalg.norm(vec))
        if distance < 30.0:
            return

        direction = vec / distance
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        stretch = clamp(distance / 360.0, 0.55, 2.4)

        layer_canvas = np.zeros_like(target)
        sample_count = 90

        for layer in range(4):
            amplitude = (7.0 + (layer * 5.5)) * stretch
            color = (255, min(255, 195 + (layer * 14)), 110 + (layer * 25))

            upper: List[Tuple[int, int]] = []
            lower: List[Tuple[int, int]] = []

            for i in range(sample_count + 1):
                t = i / sample_count
                base = p1 + (vec * t)

                envelope = 0.35 + (0.65 * math.sin(math.pi * t))
                ripple = self._noise(t + (layer * 0.02), layer)
                sway = amplitude * envelope * ripple

                depth = direction * \
                    (math.cos((t * 9.0) + self.phase + layer) * amplitude * 0.12)
                offset = normal * sway

                up = base + depth + offset
                down = base + depth - offset
                upper.append((int(up[0]), int(up[1])))
                lower.append((int(down[0]), int(down[1])))

            upper_arr = np.array(upper, dtype=np.int32)
            lower_arr = np.array(lower, dtype=np.int32)

            fill_poly = np.vstack([upper_arr, lower_arr[::-1]])
            cv2.fillPoly(layer_canvas, [fill_poly], color)
            cv2.polylines(layer_canvas, [upper_arr],
                          False, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.polylines(layer_canvas, [lower_arr],
                          False, (230, 250, 255), 1, cv2.LINE_AA)

        cv2.line(
            layer_canvas,
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )

        # Alpha blend into target to keep the bridge transparent.
        cv2.addWeighted(target, 1.0, layer_canvas, 0.28, 0.0, target)


def finger_extended(landmarks: np.ndarray, tip: int, pip: int, margin: float = 0.018) -> bool:
    return float(landmarks[tip, 1]) < float(landmarks[pip, 1]) - margin


def thumb_extended(landmarks: np.ndarray) -> bool:
    wrist = landmarks[WRIST, :2]
    tip = landmarks[THUMB_TIP, :2]
    ip = landmarks[THUMB_IP, :2]
    return float(np.linalg.norm(tip - wrist)) > float(np.linalg.norm(ip - wrist)) + 0.01


def thumb_direction(landmarks: np.ndarray) -> Optional[str]:
    tip = landmarks[THUMB_TIP, :2]
    mcp = landmarks[THUMB_MCP, :2]

    dx = float(tip[0] - mcp[0])
    dy = float(tip[1] - mcp[1])

    if abs(dy) > abs(dx) + 0.02:
        if dy < -0.04:
            return "UP"
        if dy > 0.04:
            return "DOWN"

    if abs(dx) >= abs(dy):
        if dx > 0.05:
            return "RIGHT"
        if dx < -0.05:
            return "LEFT"

    return None


def classify_hand(hand: TrackedHand) -> HandState:
    lm = hand.landmarks

    idx = finger_extended(lm, INDEX_TIP, INDEX_PIP)
    mid = finger_extended(lm, MIDDLE_TIP, MIDDLE_PIP)
    ring = finger_extended(lm, RING_TIP, RING_PIP)
    pinky = finger_extended(lm, PINKY_TIP, PINKY_PIP)
    thumb = thumb_extended(lm)

    extended_count = int(thumb) + int(idx) + int(mid) + int(ring) + int(pinky)
    fist = extended_count == 0
    open_palm = extended_count >= 4
    index_only = idx and (not mid) and (not ring) and (not pinky)
    v_sign = idx and mid and (not ring) and (not pinky)
    thumb_only = thumb and (not idx) and (
        not mid) and (not ring) and (not pinky)

    tdir = thumb_direction(lm) if thumb_only else None

    return HandState(
        label=hand.label,
        landmarks=lm,
        thumb_only=thumb_only,
        thumb_direction=tdir,
        index_extended=idx,
        middle_extended=mid,
        ring_extended=ring,
        pinky_extended=pinky,
        open_palm=open_palm,
        index_only=index_only,
        v_sign=v_sign,
        fist_closed=fist,
    )


def build_active_gestures(states: List[HandState]) -> Set[str]:
    active: Set[str] = set()

    if len(states) == 2 and states[0].index_only and states[1].index_only:
        active.add("TWO_INDEX_PLAY")

    if any(state.fist_closed for state in states):
        active.add("FIST_PAUSE")

    if any(state.v_sign for state in states):
        active.add("V_SIGN_SPOTIFY")

    for state in states:
        if not state.thumb_only or not state.thumb_direction:
            continue
        if state.thumb_direction == "UP":
            active.add("THUMB_UP_VOL")
        elif state.thumb_direction == "DOWN":
            active.add("THUMB_DOWN_VOL")
        elif state.thumb_direction == "RIGHT":
            active.add("THUMB_RIGHT_NEXT")
        elif state.thumb_direction == "LEFT":
            active.add("THUMB_LEFT_PREV")

    return active


def draw_hud(frame: np.ndarray, last_command: str, tracker_mode: str) -> None:
    lines = [
        "Hydra Gesture Map:",
        "Thumb Up/Down = Volume +/-",
        "Thumb Right/Left = Next/Previous",
        "V Sign = Open Spotify",
        "Fist = Pause | Two index-only hands = Play",
        "Two open palms = Transparent Liquid Bridge",
        f"Tracker mode: {tracker_mode}",
    ]

    if last_command:
        lines.insert(0, f"Action: {last_command}")

    y = 28
    for line in lines:
        cv2.putText(frame, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (12, 12, 12), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (235, 235, 235), 1, cv2.LINE_AA)
        y += 24


def main() -> int:
    try:
        tracker = HandTracker()
    except Exception as exc:
        print(f"Failed to initialize hand tracking: {exc}")
        return 1

    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        print("Could not open camera.")
        tracker.close()
        return 1

    visualizer = LiquidVisualizer()
    state_machine = HydraStateMachine()
    executor = CommandExecutor()
    show_hud = True

    window_name = "Hydra Water Interface"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Hydra controls:")
    print("Thumb Up/Down -> Volume")
    print("Thumb Right/Left -> Next/Previous")
    print("V sign -> Spotify")
    print("Fist -> Pause")
    print("Two index-only hands -> Play")
    print("Open palms -> Liquid bridge")
    print("H: toggle HUD, Q or ESC: quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame = cv2.flip(frame, 1)
            overlay = np.zeros_like(frame)
            now = time.perf_counter()

            tracked_hands = tracker.detect(frame)
            states = [classify_hand(hand) for hand in tracked_hands]
            states.sort(key=lambda s: float(s.wrist[0]))

            open_palms = len(
                states) == 2 and states[0].open_palm and states[1].open_palm
            if open_palms:
                visualizer.draw_water(
                    overlay, states[0].wrist, states[1].wrist)

            active = build_active_gestures(states)
            command = state_machine.resolve(active, now)
            if command:
                executor.execute(command, now)

            for state in states:
                px = int(
                    clamp(float(state.wrist[0]), 0.0, 1.0) * frame.shape[1])
                py = int(
                    clamp(float(state.wrist[1]), 0.0, 1.0) * frame.shape[0])
                tag = state.label
                if state.thumb_only and state.thumb_direction:
                    tag += f" | Thumb {state.thumb_direction}"
                elif state.v_sign:
                    tag += " | V"
                elif state.fist_closed:
                    tag += " | Fist"
                elif state.index_only:
                    tag += " | Index"
                elif state.open_palm:
                    tag += " | Palm"

                cv2.putText(
                    overlay,
                    tag,
                    (px - 60, py - 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (220, 245, 255),
                    1,
                    cv2.LINE_AA,
                )

            combined = cv2.addWeighted(frame, 1.0, overlay, 0.85, 0)
            if show_hud:
                draw_hud(combined, executor.recent_text(now), tracker.mode)
                if pyautogui is None:
                    cv2.putText(
                        combined,
                        "pyautogui missing: media keys disabled",
                        (18, combined.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (180, 220, 255),
                        1,
                        cv2.LINE_AA,
                    )

            cv2.imshow(window_name, combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("h"):
                show_hud = not show_hud

    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
