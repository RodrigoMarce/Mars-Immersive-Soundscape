from collections import deque
from typing import List

from helpers import iou_xyxy


class TrackManager:
    """Manage person tracks: match detections, create/update tracks, remove stale tracks."""

    def __init__(self, max_misses: int = 15, hist_len: int = 20, iou_threshold: float = 0.3):
        self.tracks = {}  # tid -> { bbox, hist, last_seen }
        self.next_id = 1
        self.max_misses = max_misses
        self.hist_len = hist_len
        self.iou_threshold = iou_threshold

    def update(self, detections: List, frame_idx: int):
        """Match detections to existing tracks and create/update tracks."""
        used_tracks = set()
        assigned = []

        for box in detections:
            best_id = None
            best_iou = 0
            for tid, t in self.tracks.items():
                if tid in used_tracks:
                    continue
                score = iou_xyxy(box, t["bbox"])
                if score > best_iou:
                    best_iou = score
                    best_id = tid

            if best_iou > self.iou_threshold:
                assigned.append((best_id, box))
                used_tracks.add(best_id)
            else:
                assigned.append((None, box))

        for tid, box in assigned:
            if tid is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "bbox": box,
                    "hist": deque(maxlen=self.hist_len),
                    "last_seen": frame_idx,
                }
            else:
                self.tracks[tid]["bbox"] = box
                self.tracks[tid]["last_seen"] = frame_idx

    def remove_stale(self, frame_idx: int) -> List[int]:
        """Remove tracks not seen for more than `max_misses`. Returns list of removed ids."""
        removed = []
        for tid in list(self.tracks.keys()):
            if frame_idx - self.tracks[tid]["last_seen"] > self.max_misses:
                del self.tracks[tid]
                removed.append(tid)
        return removed

    def get_tracks(self):
        return self.tracks
