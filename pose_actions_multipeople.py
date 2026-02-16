import time
from collections import deque

import cv2
from ultralytics import YOLO

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from osc_client import OSCClient

from helpers import (
    clamp_box,
    expand_box,
    energy_to_level,
    compute_energy,
    compute_group_stats,
    average_to_level,
    max_energy_converter,
)
from tracker import TrackManager


# ------------------ OSC ------------------

OSC_IP = "127.0.0.1"
OSC_PORT = 8000


def run_detector(frame, detector):
    """Run YOLO detector on frame and return person bounding boxes (xyxy)."""
    det = detector(frame, verbose=False)[0]
    detections = []
    for box, cls in zip(det.boxes.xyxy.cpu().numpy(), det.boxes.cls.cpu().numpy()):
        if int(cls) == 0:
            detections.append(box)
    return detections


def main():
    osc = OSCClient(OSC_IP, OSC_PORT)

    detector = YOLO("yolov8n.pt")

    base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
    )
    pose = vision.PoseLandmarker.create_from_options(options)

    MAX_MISSES = 15

    tracker = TrackManager(MAX_MISSES)
    prev_people_count = -1
    # person_energy_state maps tid -> { "level": str, "value": float }
    person_energy_state = {}

    # last sent values for group stats (floats)
    last_sent_avg = None
    last_sent_max = None
    last_sent_std_time = 0.0

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

        t_now = time.time()
        dt = t_now - t_prev
        t_prev = t_now
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))

        detections = run_detector(frame, detector)

        tracker.update(detections, frame_idx)

        removed = tracker.remove_stale(frame_idx)
        for tid in removed:
            if tid in person_energy_state:
                del person_energy_state[tid]

        tracks = tracker.get_tracks()

        people_count = len(tracks)

        # If people count changes, send OSC and print everyone's energy levels
        if people_count != prev_people_count:
            osc.send_people(people_count)
            # Print energy level list (strings) for everyone currently known
            energy_list = [v["level"] for v in person_energy_state.values()]
            print("People changed — energy levels:", energy_list)
            prev_people_count = people_count

        # Per-person pose + energy
        for tid, t in tracks.items():
            x1, y1, x2, y2 = expand_box(*t["bbox"], W, H)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
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
                "rh": p(24),
            }

            t["hist"].append(keypoints)

            _, _, _, norm = compute_energy(t["hist"], fps)
            level = energy_to_level(norm)

            if tid not in person_energy_state:
                person_energy_state[tid] = {"level": level, "value": norm}
                osc.send_energy(tid, level)
            else:
                prev_level = person_energy_state[tid]["level"]
                if level != prev_level:
                    osc.send_energy(tid, level)
                # Always update stored value
                person_energy_state[tid] = {"level": level, "value": norm}

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {tid}  {level}",
                (int(x1), int(y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # Compute group stats (numeric floats 0..1)
        numeric_vals = [v["value"] for v in person_energy_state.values()]
        avg, mn, mx, std = compute_group_stats(numeric_vals)

        # Send OSC when avg/max change by more than 0.1
        if avg is not None:
            if last_sent_avg is None or abs(avg - last_sent_avg) > 0.1:
                osc.send_group_avg(avg)
                last_sent_avg = avg

        if mx is not None:
            if last_sent_max is None or abs(mx - last_sent_max) > 0.1:
                osc.send_group_max(mx)
                last_sent_max = mx

        # Send std every 1 second
        now_t = time.time()
        if std is not None and (now_t - last_sent_std_time) >= 1.0:
            osc.send_group_std(std)
            last_sent_std_time = now_t

        # Convert numeric stats to level strings for display
        avg_level = average_to_level(avg)
        max_level = max_energy_converter(mx)

        # Overlay FPS, people, and group energy stats
        if avg is None:
            stats_text = f"FPS: {fps:.1f}   People: {people_count}"
        else:
            stats_text = (
                f"FPS: {fps:.1f}   People: {people_count}   "
                f"Avg: {avg:.2f}({avg_level})  Max: {mx:.2f}({max_level})  Std: {std:.2f}"
            )

        cv2.putText(
            frame,
            stats_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Pose Energy", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
