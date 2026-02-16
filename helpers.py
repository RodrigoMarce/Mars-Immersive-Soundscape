import numpy as np


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


def compute_energy(hist, fps):
    """Compute arm, leg, core and normalized energy from a history deque.

    Returns: (arms, legs, core, normalized)
    """
    if len(hist) < 6:
        return 0, 0, 0, 0

    dt = 1.0 / max(10.0, fps)

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


def compute_group_stats(values):
    """Compute avg, min, max, stddev for a list of numeric energy values.

    Returns (avg, mn, mx, std). If `values` is empty, returns (None, None, None, None).
    """
    if not values:
        return None, None, None, None
    vals = np.array(values, dtype=float)
    avg = float(np.mean(vals))
    mn = float(np.min(vals))
    mx = float(np.max(vals))
    std = float(np.std(vals))
    return avg, mn, mx, std


def average_to_level(avg_value):
    """Map an average numeric energy value to a level string using same thresholds.

    This reuses the same thresholds as `energy_to_level`.
    """
    if avg_value is None:
        return None
    return energy_to_level(avg_value)


def max_energy_converter(max_value):
    """TODO: Map a max numeric energy value to a level string using same thresholds.
    """
    if max_value is None:
        return None
    return energy_to_level(max_value)  # Change this to perform something random maybe
