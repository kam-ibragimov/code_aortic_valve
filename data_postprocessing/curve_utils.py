import numpy as np


def _resample_curve(pts, n_points):
    diffs = np.diff(pts, axis=0)
    cum_len = np.concatenate([[0], np.cumsum(np.linalg.norm(diffs, axis=1))])
    target = np.linspace(0, cum_len[-1], n_points)
    resampled = np.column_stack([np.interp(target, cum_len, pts[:, i]) for i in range(3)])
    return resampled.tolist()


def _sample_closed_curve_uniform(points, n_points):
    """Sample n_points equally spaced (by arc length) on a closed curve.
    The wrap-around segment (last→first) is included so all N gaps are equal.
    Returns list of [x, y, z] — first and last points are distinct.
    """
    pts = np.asarray(points, dtype=float)
    pts_closed = np.vstack([pts, pts[0:1]])          # close the loop for arc-length calc
    seg_lengths = np.linalg.norm(np.diff(pts_closed, axis=0), axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cum_len[-1]
    if total_length == 0 or len(pts) < 2:
        return None
    targets = np.arange(n_points) * (total_length / n_points)
    resampled = np.column_stack([np.interp(targets, cum_len, pts_closed[:, i]) for i in range(3)])
    return resampled.tolist()


def point_to_curve_distances(gt_points, curve_points):
    """Nearest-point distance from each GT point to curve_points."""
    gt = np.asarray(gt_points, dtype=float)
    curve = np.asarray(curve_points, dtype=float)
    return [np.sqrt(np.sum((curve - p) ** 2, axis=1)).min() for p in gt]


def mean_point_to_curve_distance(gt_points, curve_points):
    """Mean point-to-curve distance (MPCD): for each GT point, the distance to
    the nearest point on curve_points, averaged over all GT points."""
    return float(np.mean(point_to_curve_distances(gt_points, curve_points)))