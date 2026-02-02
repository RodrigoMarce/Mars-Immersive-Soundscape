import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

import mediapipe as mp
mp_pose = mp.solutions.pose
LM = mp_pose.PoseLandmark

KEYS = {
    "lw": LM.LEFT_WRIST.value,
    "rw": LM.RIGHT_WRIST.value,
    "la": LM.LEFT_ANKLE.value,
    "ra": LM.RIGHT_ANKLE.value,
    "lh": LM.LEFT_HIP.value,
    "rh": LM.RIGHT_HIP.value,
    "ls": LM.LEFT_SHOULDER.value,
    "rs": LM.RIGHT_SHOULDER.value,
}

# -----------------------------
# Helpers
# -----------------------------
def clamp_box(x1, y1, x2, y2, w, h):
    x1 = int(max(0, min(x1, w - 1)))
    y1 = int(max(0, min(y1, h - 1)))
    x2 = int(max(0, min(x2, w - 1)))
    y2 = int(max(0, min(y2, h - 1)))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def expand_box(x1, y1, x2, y2, w, h, scale=1.6):
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = (x2 - x1) * scale, (y2 - y1) * scale
    nx1, ny1 = cx - bw / 2.0, cy - bh / 2.0
    nx2, ny2 = cx + bw / 2.0, cy + bh / 2.0
    return clamp_box(nx1, ny1, nx2, ny2, w, h)


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return float(inter / union)


def majority_vote(labels):
    if not labels:
        return "..."
    vals, counts = np.unique(np.array(labels), return_counts=True)
    return vals[np.argmax(counts)]


# -----------------------------
# Pose landmarks we’ll use
# (MediaPipe Pose indices)
# -----------------------------


def extract_keypoints(results):
    """Return dict of normalized (x,y,vis) for key landmarks, or None if not present."""
    if not results.pose_landmarks:
        return None
    pts = results.pose_landmarks.landmark
    out = {}
    for name, idx in KEYS.items():
        out[name] = (pts[idx].x, pts[idx].y, pts[idx].visibility)
    return out


def vec2(a, b):
    return np.array([b[0] - a[0], b[1] - a[1]], dtype=np.float32)


# -----------------------------
# Heuristic motion classifier from keypoints
# -----------------------------
def classify_from_history(hist, fps_est):
    """
    hist: deque of dict keypoints over time (normalized coords in crop)
    fps_est: estimated FPS for velocity scaling (can be rough)
    Returns label string.
    """
    if len(hist) < 8:
        return "..."

    # Only use frames with decent visibility on core joints
    def good(frame):
        return (
            frame["lh"][2] > 0.4 and frame["rh"][2] > 0.4 and
            frame["la"][2] > 0.4 and frame["ra"][2] > 0.4 and
            frame["lw"][2] > 0.3 and frame["rw"][2] > 0.3
        )

    frames = [f for f in hist if good(f)]
    if len(frames) < 6:
        return "..."

    # Build velocities (normalized units per second)
    dt = 1.0 / max(10.0, float(fps_est))  # avoid crazy dt
    def speed(series_xy):
        # series_xy: list of (x,y)
        v = []
        for i in range(1, len(series_xy)):
            d = np.linalg.norm(np.array(series_xy[i]) - np.array(series_xy[i-1]))
            v.append(d / dt)
        return np.array(v, dtype=np.float32)

    lw = [(f["lw"][0], f["lw"][1]) for f in frames]
    rw = [(f["rw"][0], f["rw"][1]) for f in frames]
    la = [(f["la"][0], f["la"][1]) for f in frames]
    ra = [(f["ra"][0], f["ra"][1]) for f in frames]
    lh = [(f["lh"][0], f["lh"][1]) for f in frames]
    rh = [(f["rh"][0], f["rh"][1]) for f in frames]

    lw_s = speed(lw)
    rw_s = speed(rw)
    ankle_s = 0.5 * (speed(la) + speed(ra))
    hip_s = 0.5 * (speed(lh) + speed(rh))

    # Vertical ankle velocity (jump cue): ankle y changes fast up/down
    la_y = np.array([p[1] for p in la], dtype=np.float32)
    ra_y = np.array([p[1] for p in ra], dtype=np.float32)
    vy = np.abs(np.diff(0.5 * (la_y + ra_y)) / dt)  # abs vertical speed

    # Overall motion energy: weighted sum of joint speeds
    arms_energy = float(np.median(lw_s + rw_s))
    legs_energy = float(np.median(ankle_s))
    core_energy = float(np.median(hip_s))
    total_energy = 0.45 * arms_energy + 0.45 * legs_energy + 0.10 * core_energy

    # Rhythm/variability: dancing tends to have high total energy + higher variance
    total_var = float(np.var(ankle_s) + np.var(lw_s + rw_s))

    # Idle threshold (tune)
    if total_energy < 1.2:
        return "idle"

    # Jumping: strong vertical ankle speed spikes
    if np.percentile(vy, 85) > 6.0 and legs_energy > 2.0:
        return "jumping"

    # Running vs walking (cadence proxy): ankle speed magnitude
    # (These thresholds are heuristic; you will likely tune them.)
    if legs_energy > 4.0 and core_energy > 1.5:
        return "running"
    if legs_energy > 2.0 and core_energy > 1.0:
        return "walking"

    # Arms moving (without strong leg motion)
    if arms_energy > 3.0 and legs_energy < 2.0:
        return "arms_moving"

    # Dancing: high total motion + variability across body
    if total_energy > 2.5 and total_var > 6.0:
        return "dancing"

    # Fallback for “motion actions”
    return "moving"


# -----------------------------
# Main
# -----------------------------
def main():
    # Detector
    detector = YOLO("yolov8n.pt")
    PERSON_CLASS_ID = 0
    CONF_THRESH = 0.25

    # Pose model (single-person) — we run it on each person crop
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,           # try 0 for speed, 2 for accuracy
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Tracking
    tracks = {}  # id -> dict(bbox, last_seen, hist, labels_hist)
    next_id = 1
    IOU_THRESH = 0.3
    MAX_MISSES = 20

    # History sizes
    HIST_LEN = 30        # ~1 sec at 30 FPS (enough for motion)
    LABEL_SMOOTH = 10

    # Performance knobs
    INFER_POSE_EVERY_N_FRAMES = 2   # run pose every N frames per track (increase if slow)
    MAX_POSE_PEOPLE = 6             # only run pose on top-K biggest boxes (raise/lower as needed)
    BOX_EXPAND = 1.7                # include context for better motion cues

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    frame_idx = 0
    fps_est = 30.0
    t_prev = time.time()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        H, W = frame_bgr.shape[:2]

        # FPS estimate
        t_now = time.time()
        dt = max(1e-6, t_now - t_prev)
        t_prev = t_now
        fps_est = 0.9 * fps_est + 0.1 * (1.0 / dt)

        # Detect people
        det = detector.predict(frame_bgr, imgsz=640, conf=CONF_THRESH, verbose=False)[0]
        detections = []
        if det.boxes is not None and len(det.boxes) > 0:
            xyxy = det.boxes.xyxy.cpu().numpy()
            cls = det.boxes.cls.cpu().numpy().astype(int)
            conf = det.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
                if c == PERSON_CLASS_ID:
                    detections.append((float(x1), float(y1), float(x2), float(y2), float(p)))

        # Match detections to tracks (greedy IoU)
        used = set()
        assigned = []
        for (x1, y1, x2, y2, p) in detections:
            best_id, best_iou = None, 0.0
            for tid, t in tracks.items():
                if tid in used:
                    continue
                score = iou_xyxy((x1, y1, x2, y2), t["bbox"])
                if score > best_iou:
                    best_iou, best_id = score, tid
            if best_id is not None and best_iou >= IOU_THRESH:
                used.add(best_id)
                assigned.append((best_id, (x1, y1, x2, y2), p))
            else:
                assigned.append((None, (x1, y1, x2, y2), p))

        # Update/create tracks
        for tid, (x1, y1, x2, y2), p in assigned:
            if tid is None:
                tid = next_id
                next_id += 1
                tracks[tid] = {
                    "bbox": (x1, y1, x2, y2),
                    "last_seen": frame_idx,
                    "hist": deque(maxlen=HIST_LEN),
                    "labels": deque(maxlen=LABEL_SMOOTH),
                    "label_now": "...",
                    "conf_now": p,
                }
            else:
                tracks[tid]["bbox"] = (x1, y1, x2, y2)
                tracks[tid]["last_seen"] = frame_idx
                tracks[tid]["conf_now"] = p

        # Remove stale tracks
        for tid in list(tracks.keys()):
            if frame_idx - tracks[tid]["last_seen"] > MAX_MISSES:
                del tracks[tid]

        # Choose which tracks to run pose on (top-K by area)
        track_items = []
        for tid, t in tracks.items():
            x1, y1, x2, y2 = t["bbox"]
            area = max(0.0, (x2 - x1) * (y2 - y1))
            track_items.append((area, tid))
        track_items.sort(reverse=True)
        pose_tids = [tid for _, tid in track_items[:MAX_POSE_PEOPLE]]

        # Pose inference (per crop)
        if frame_idx % INFER_POSE_EVERY_N_FRAMES == 0:
            for tid in pose_tids:
                t = tracks[tid]
                x1, y1, x2, y2 = t["bbox"]
                ex1, ey1, ex2, ey2 = expand_box(x1, y1, x2, y2, W, H, scale=BOX_EXPAND)
                crop_bgr = frame_bgr[ey1:ey2, ex1:ex2]
                if crop_bgr.size == 0:
                    continue

                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                results = pose.process(crop_rgb)
                kp = extract_keypoints(results)
                if kp is None:
                    continue

                # Add keypoints to history
                t["hist"].append(kp)

                # Classify from history
                label = classify_from_history(t["hist"], fps_est)
                t["labels"].append(label)
                t["label_now"] = majority_vote(list(t["labels"]))

        # Draw
        out = frame_bgr.copy()
        for tid, t in tracks.items():
            x1, y1, x2, y2 = t["bbox"]
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H)

            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            txt = f"ID {tid}: {t['label_now']}"
            y_text = max(20, y1 - 10)
            cv2.putText(out, txt, (x1, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(out, f"FPS~{fps_est:.0f}  Tracks={len(tracks)}  PoseTopK={MAX_POSE_PEOPLE}",
                    (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Multi-person Pose Actions (q to quit)", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    main()
