import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# Pose Landmarker (NEW API)
# -----------------------------
base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)


# BlazePose landmark indices
KEYS = {
    "lw": 15,
    "rw": 16,
    "la": 27,
    "ra": 28,
    "lh": 23,
    "rh": 24,
}


# -----------------------------
# Utilities
# -----------------------------
def clamp_box(x1, y1, x2, y2, w, h):
    return (
        int(max(0, min(x1, w - 1))),
        int(max(0, min(y1, h - 1))),
        int(max(0, min(x2, w - 1))),
        int(max(0, min(y2, h - 1))),
    )


def expand_box(x1, y1, x2, y2, w, h, scale=1.6):
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * scale, (y2 - y1) * scale
    return clamp_box(cx - bw / 2, cy - bh / 2,
                     cx + bw / 2, cy + bh / 2, w, h)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    return inter / (area_a + area_b - inter + 1e-6)


# -----------------------------
# Energy Computation
# -----------------------------
def compute_energy(hist, fps):
    if len(hist) < 6:
        return None

    dt = 1.0 / max(fps, 10)

    def speed(points):
        v = []
        for i in range(1, len(points)):
            d = np.linalg.norm(
                np.array(points[i]) - np.array(points[i - 1])
            )
            v.append(d / dt)
        return np.array(v)

    lw = [f["lw"] for f in hist]
    rw = [f["rw"] for f in hist]
    la = [f["la"] for f in hist]
    ra = [f["ra"] for f in hist]
    lh = [f["lh"] for f in hist]
    rh = [f["rh"] for f in hist]

    arms_speed = np.median(speed(lw) + speed(rw))
    legs_speed = np.median(speed(la) + speed(ra))
    core_speed = np.median(speed(lh) + speed(rh))

    total = 0.45 * arms_speed + 0.45 * legs_speed + 0.10 * core_speed

    normalized = float(np.clip(total / 3, 0, 1))

    return {
        "total": normalized,
        "arms": float(arms_speed),
        "legs": float(legs_speed),
        "core": float(core_speed),
    }


# -----------------------------
# Main
# -----------------------------
def main():
    detector = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not accessible")

    tracks = {}
    next_id = 1
    IOU_THRESH = 0.3
    MAX_MISS = 20

    fps_est = 30
    t_prev = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        H, W = frame.shape[:2]

        # FPS estimation
        t_now = time.time()
        dt = t_now - t_prev
        t_prev = t_now
        fps_est = 0.9 * fps_est + 0.1 * (1.0 / max(dt, 1e-6))

        # Detection
        result = detector.predict(frame, conf=0.25, verbose=False)[0]

        detections = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(boxes, classes):
                if int(c) == 0:
                    detections.append((x1, y1, x2, y2))

        # Matching
        used = set()
        for box in detections:
            best_id = None
            best_iou = 0
            for tid, t in tracks.items():
                if tid in used:
                    continue
                score = iou(box, t["bbox"])
                if score > best_iou:
                    best_iou = score
                    best_id = tid

            if best_iou > IOU_THRESH:
                tracks[best_id]["bbox"] = box
                tracks[best_id]["last"] = frame_idx
                used.add(best_id)
            else:
                tracks[next_id] = {
                    "bbox": box,
                    "last": frame_idx,
                    "hist": deque(maxlen=30),
                    "energy": None,
                }
                next_id += 1

        # Remove stale tracks
        for tid in list(tracks.keys()):
            if frame_idx - tracks[tid]["last"] > MAX_MISS:
                del tracks[tid]

        # Pose + Energy
        for tid, t in tracks.items():
            x1, y1, x2, y2 = expand_box(*t["bbox"], W, H, 1.6)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = pose_landmarker.detect(mp_image)

            if not result.pose_landmarks:
                continue

            landmarks = result.pose_landmarks[0]

            kp = {}
            for name, idx in KEYS.items():
                lm = landmarks[idx]
                kp[name] = (lm.x, lm.y)

            t["hist"].append(kp)
            t["energy"] = compute_energy(t["hist"], fps_est)

        # Drawing
        for tid, t in tracks.items():
            x1, y1, x2, y2 = clamp_box(*t["bbox"], W, H)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if t["energy"] is not None:
                e = t["energy"]

                text_lines = [
                    f"ID {tid}",
                    f"Total: {e['total']:.2f}",
                    f"Arms: {e['arms']:.2f}",
                    f"Legs: {e['legs']:.2f}",
                    f"Core: {e['core']:.2f}",
                    f"Hist: {len(t['hist'])}",
                ]

                for i, txt in enumerate(text_lines):
                    cv2.putText(frame, txt,
                                (x1, y1 - 10 - i * 18),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2)

        cv2.putText(frame, f"FPS: {fps_est:.1f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 255), 2)

        cv2.imshow("Pose Energy Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose_landmarker.close()


if __name__ == "__main__":
    main()
