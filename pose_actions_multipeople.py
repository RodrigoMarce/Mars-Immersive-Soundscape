import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pythonosc.udp_client import SimpleUDPClient


# ------------------ OSC ------------------

OSC_IP = "127.0.0.1"
OSC_PORT = 8000
osc = SimpleUDPClient(OSC_IP, OSC_PORT)


# ------------------ Helpers ------------------

def clamp_box(x1, y1, x2, y2, w, h):
    x1 = int(max(0, min(x1, w - 1)))
    y1 = int(max(0, min(y1, h - 1)))
    x2 = int(max(0, min(x2, w - 1)))
    y2 = int(max(0, min(y2, h - 1)))
    return x1, y1, x2, y2


def expand_box(x1, y1, x2, y2, w, h, scale=1.6):
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = (x2 - x1) * scale, (y2 - y1) * scale
    nx1, ny1 = cx - bw / 2.0, cy - bh / 2.0
    nx2, ny2 = cx + bw / 2.0, cy + bh / 2.0
    return clamp_box(nx1, ny1, nx2, ny2, w, h)


def iou_xyxy(a, b):
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
    union = area_a + area_b - inter + 1e-6
    return inter / union


def energy_to_level(e):
    if e < 0.1:
        return "idle"
    elif e < 0.35:
        return "light"
    elif e < 0.7:
        return "moderate"
    else:
        return "high"


# ------------------ Energy ------------------

def compute_energy(hist, fps):
    if len(hist) < 6:
        return 0, 0, 0, 0

    dt = 1.0 / max(10.0, fps)   # FPS is low. Updates are infrequent

    def speed(series):
        return [
            np.linalg.norm(np.array(series[i]) - np.array(series[i - 1])) / dt
            for i in range(1, len(series))
        ]

    lw = [h["lw"] for h in hist]
    rw = [h["rw"] for h in hist]
    la = [h["la"] for h in hist]
    ra = [h["ra"] for h in hist]
    lh = [h["lh"] for h in hist]
    rh = [h["rh"] for h in hist]

    arms = np.median(speed(lw) + speed(rw))
    legs = np.median(speed(la) + speed(ra))
    core = np.median(speed(lh) + speed(rh))

    total = 0.45 * arms + 0.45 * legs + 0.10 * core
    normalized = np.clip(total / 3.0, 0.0, 1.0)

    return arms, legs, core, normalized


# ------------------ Main ------------------

def main():

    detector = YOLO("yolov8n.pt")

    base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE
    )
    pose = vision.PoseLandmarker.create_from_options(options)

    tracks = {}
    next_id = 1
    prev_people_count = -1
    person_energy_state = {}

    MAX_MISSES = 15  # If soneone is not seen for these many frames, delete track
    frame_idx = 0

    cap = cv2.VideoCapture(0)

    fps = 30
    t_prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        H, W = frame.shape[:2]

        # FPS
        t_now = time.time()
        dt = t_now - t_prev
        t_prev = t_now
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))

        # Detect
        det = detector(frame, verbose=False)[0] # on for debugging
        detections = []

        for box, cls in zip(det.boxes.xyxy.cpu().numpy(),
                            det.boxes.cls.cpu().numpy()):
            if int(cls) == 0:
                detections.append(box)

        # -------- Matching --------
        used_tracks = set()
        assigned = []

        for box in detections:
            best_id = None
            best_iou = 0
            for tid, t in tracks.items():
                if tid in used_tracks:
                    continue
                score = iou_xyxy(box, t["bbox"])
                if score > best_iou:
                    best_iou = score
                    best_id = tid

            if best_iou > 0.3:  # IoU threshold. Lower = more likely to have multiple IDs for same person
                assigned.append((best_id, box))
                used_tracks.add(best_id)
            else:
                assigned.append((None, box))

        # -------- Update/Create --------
        for tid, box in assigned:
            if tid is None:
                tid = next_id
                next_id += 1
                tracks[tid] = {
                    "bbox": box,
                    "hist": deque(maxlen=20),  # number of frames to keep in history
                    "last_seen": frame_idx
                }
            else:
                tracks[tid]["bbox"] = box
                tracks[tid]["last_seen"] = frame_idx

        # -------- Remove stale tracks --------
        for tid in list(tracks.keys()):
            if frame_idx - tracks[tid]["last_seen"] > MAX_MISSES:
                del tracks[tid]
                if tid in person_energy_state:
                    del person_energy_state[tid]

        people_count = len(tracks)

        # OSC: people changed
        if people_count != prev_people_count:
            osc.send_message("/people", people_count)
            prev_people_count = people_count

        # -------- Pose per person --------
        for tid, t in tracks.items():

            x1, y1, x2, y2 = expand_box(*t["bbox"], W, H)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            )

            result = pose.detect(mp_image)

            if not result.pose_landmarks:
                continue

            lm = result.pose_landmarks[0]

            def p(i):
                return (lm[i].x, lm[i].y)

            keypoints = {
                "lw": p(15),
                "rw": p(16),
                "la": p(27),
                "ra": p(28),
                "lh": p(23),
                "rh": p(24)
            }

            t["hist"].append(keypoints)

            _, _, _, norm = compute_energy(t["hist"], fps)  # bind these for debugging
            level = energy_to_level(norm)

            if tid not in person_energy_state:
                person_energy_state[tid] = level
                osc.send_message("/energy", [tid, level])
            else:
                if level != person_energy_state[tid]:
                    osc.send_message("/energy", [tid, level])
                    person_energy_state[tid] = level

            cv2.rectangle(frame,
                          (int(x1), int(y1)),
                          (int(x2), int(y2)),
                          (0, 255, 0),
                          2)

            cv2.putText(frame,
                        f"ID {tid}  {level}",
                        (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

        # Overlay FPS + people
        cv2.putText(frame,
                    f"FPS: {fps:.1f}   People: {people_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2)

        cv2.imshow("Pose Energy", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
