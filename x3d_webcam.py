"""
Multi-person realtime action recognition:
- Detect people with YOLOv8
- Track them with a simple IoU matcher (assign stable IDs)
- Run X3D-S action recognition per person using per-track clip buffers
- Draw a box and label for each person

Fix included:
- Person crops are resized/letterboxed to a fixed size before buffering
  to avoid np.stack shape mismatch.

Install:
  pip install torch opencv-python numpy ultralytics

Run:
  python x3d_multipeople.py
"""

import json
import time
import urllib.request
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO


# -----------------------------
# Kinetics labels
# -----------------------------
def download_kinetics_labels(path="kinetics_classnames.json"):
    url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as e:
        raise RuntimeError(f"Failed to download labels: {e}")

    with open(path, "r") as f:
        name_to_idx = json.load(f)

    idx_to_name = {int(idx): str(name) for name, idx in name_to_idx.items()}
    return idx_to_name


# -----------------------------
# Image helpers
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


def letterbox_to_square(img_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    """
    Resize keeping aspect ratio, pad to (size,size).
    Returns uint8 BGR image of shape (size, size, 3).
    """
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)

    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


# -----------------------------
# Video preprocessing (no torchvision)
# -----------------------------
def resize_short_side(video_tchw: torch.Tensor, short_side: int) -> torch.Tensor:
    # video_tchw: (T, C, H, W)
    T, C, H, W = video_tchw.shape
    if min(H, W) == short_side:
        return video_tchw
    scale = short_side / float(min(H, W))
    new_h = int(round(H * scale))
    new_w = int(round(W * scale))
    return F.interpolate(video_tchw, size=(new_h, new_w), mode="bilinear", align_corners=False)


def center_crop(video_tchw: torch.Tensor, crop_size: int) -> torch.Tensor:
    # video_tchw: (T, C, H, W)
    T, C, H, W = video_tchw.shape
    if H < crop_size or W < crop_size:
        video_tchw = resize_short_side(video_tchw, crop_size)
        T, C, H, W = video_tchw.shape
    y0 = (H - crop_size) // 2
    x0 = (W - crop_size) // 2
    return video_tchw[:, :, y0 : y0 + crop_size, x0 : x0 + crop_size]


def uniform_temporal_subsample(frames: list, num_frames: int):
    N = len(frames)
    if N < num_frames:
        return None
    idxs = np.linspace(0, N - 1, num_frames).astype(int)
    return [frames[i] for i in idxs]


def preprocess_clip(
    frames_bgr: list,
    num_frames=13,
    side_size=182,
    crop_size=182,
    mean=(0.45, 0.45, 0.45),
    std=(0.225, 0.225, 0.225),
):
    """
    frames_bgr must all be same shape (H,W,3). We ensure that by letterboxing
    the person crop before adding to the buffer.
    """
    sampled = uniform_temporal_subsample(frames_bgr, num_frames)
    if sampled is None:
        return None

    # Now safe: all frames same shape
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in sampled]
    clip = np.stack(rgb, axis=0).astype(np.float32) / 255.0  # (T, H, W, C)

    video = torch.from_numpy(clip).permute(0, 3, 1, 2)  # (T, C, H, W)

    # Resize short side + center crop to match X3D hub defaults
    video = resize_short_side(video, side_size)
    video = center_crop(video, crop_size)

    mean_t = torch.tensor(mean, dtype=video.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=video.dtype).view(1, 3, 1, 1)
    video = (video - mean_t) / std_t

    # (1, C, T, H, W)
    video = video.permute(1, 0, 2, 3).unsqueeze(0)
    return video


# -----------------------------
# Simple IoU tracker
# -----------------------------
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


# -----------------------------
# Main
# -----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load X3D-S
    model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_s", pretrained=True)
    model = model.to(device).eval()
    softmax = torch.nn.Softmax(dim=1)

    idx_to_label = download_kinetics_labels()

    # Person detector
    detector = YOLO("yolov8n.pt")  # fast
    PERSON_CLASS_ID = 0

    # X3D params (hub defaults)
    NUM_FRAMES = 13
    SAMPLING_RATE = 6
    SIDE_SIZE = 182
    CROP_SIZE = 182
    clip_len_frames = NUM_FRAMES * SAMPLING_RATE  # 78

    # Make buffered crops consistent size (fixes your error)
    CROP_BUFFER_SIZE = 224  # must be constant

    # Tracking state
    tracks = {}  # id -> dict(bbox, buf, label, conf, last_seen)
    next_id = 1
    MAX_MISSES = 20
    IOU_THRESH = 0.3

    INFER_EVERY_N_FRAMES = 12
    CONF_THRESH = 0.25

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    frame_idx = 0
    last_infer_ms = 0.0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        H, W = frame_bgr.shape[:2]

        # ---- Detect persons ----
        det = detector.predict(frame_bgr, imgsz=640, conf=CONF_THRESH, verbose=False)[0]
        boxes = []
        if det.boxes is not None and len(det.boxes) > 0:
            xyxy = det.boxes.xyxy.cpu().numpy()
            cls = det.boxes.cls.cpu().numpy().astype(int)
            conf = det.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
                if c == PERSON_CLASS_ID:
                    boxes.append((float(x1), float(y1), float(x2), float(y2), float(p)))

        # ---- Match detections to existing tracks ----
        used_tracks = set()
        assigned = []

        for bx in boxes:
            x1, y1, x2, y2, p = bx
            best_id = None
            best_iou = 0.0
            for tid, t in tracks.items():
                if tid in used_tracks:
                    continue
                score = iou_xyxy((x1, y1, x2, y2), t["bbox"])
                if score > best_iou:
                    best_iou = score
                    best_id = tid
            if best_id is not None and best_iou >= IOU_THRESH:
                used_tracks.add(best_id)
                assigned.append((best_id, (x1, y1, x2, y2), p))
            else:
                assigned.append((None, (x1, y1, x2, y2), p))

        # ---- Update/create tracks + append fixed-size crops ----
        for tid, (x1, y1, x2, y2), p in assigned:
            if tid is None:
                tid = next_id
                next_id += 1
                tracks[tid] = {
                    "bbox": (x1, y1, x2, y2),
                    "buf": deque(maxlen=clip_len_frames),
                    "label": "...",
                    "conf": 0.0,
                    "last_seen": frame_idx,
                }
            else:
                tracks[tid]["bbox"] = (x1, y1, x2, y2)
                tracks[tid]["last_seen"] = frame_idx

            cx1, cy1, cx2, cy2 = clamp_box(x1, y1, x2, y2, W, H)
            crop = frame_bgr[cy1:cy2, cx1:cx2].copy()

            # FIX: normalize crop size BEFORE buffering
            crop_fixed = letterbox_to_square(crop, size=CROP_BUFFER_SIZE)
            tracks[tid]["buf"].append(crop_fixed)

        # ---- Remove stale tracks ----
        to_delete = []
        for tid, t in tracks.items():
            if frame_idx - t["last_seen"] > MAX_MISSES:
                to_delete.append(tid)
        for tid in to_delete:
            del tracks[tid]

        # ---- Inference per track ----
        if frame_idx % INFER_EVERY_N_FRAMES == 0:
            t0 = time.time()
            for tid, t in tracks.items():
                if len(t["buf"]) < clip_len_frames:
                    continue
                inputs = preprocess_clip(
                    list(t["buf"]),
                    num_frames=NUM_FRAMES,
                    side_size=SIDE_SIZE,
                    crop_size=CROP_SIZE,
                )
                if inputs is None:
                    continue
                with torch.no_grad():
                    logits = model(inputs.to(device))
                    probs = softmax(logits)[0]
                    conf, idx = torch.max(probs, dim=0)
                t["label"] = idx_to_label.get(int(idx.item()), f"class_{int(idx.item())}")
                t["conf"] = float(conf.item())
            last_infer_ms = (time.time() - t0) * 1000.0

        # ---- Draw ----
        out = frame_bgr.copy()
        for tid, t in tracks.items():
            x1, y1, x2, y2 = t["bbox"]
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H)

            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"ID {tid}: {t['label']} ({t['conf']:.2f})"
            y_text = max(20, y1 - 10)
            cv2.putText(out, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(out, f"Infer: {last_infer_ms:.0f}ms", (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("X3D-S per-person (press q to quit)", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
