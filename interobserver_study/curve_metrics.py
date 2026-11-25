import numpy as np
from scipy.interpolate import splprep, splev

CURVE_STEP_MM = 0.1
N_SPLINE_EVAL = 3000  # spline evaluation density before resampling to CURVE_STEP_MM


def _fit_spline(points, per, n_dense):
    pts = np.asarray(points, dtype=float)
    k = min(3, len(pts) - 1)
    tck, _ = splprep(pts.T, s=0, k=k, per=per)
    u = np.linspace(0, 1, n_dense)
    return np.array(splev(u, tck)).T


def fit_closed_spline(points, n_dense):
    """Exact periodic spline through ordered contour points. The contour is closed explicitly:
    markups do not repeat the first point, and splprep(per=1) would otherwise overwrite the
    last point with it, dropping that point from the contour."""
    pts = np.asarray(points, dtype=float)
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
        pts = np.vstack([pts, pts[:1]])
    return _fit_spline(pts, per=1, n_dense=n_dense)


def _resample_by_step(polyline, step):
    """Equally spaced points along the polyline with spacing as close to `step` as possible.
    Both ends are kept, so a closed polyline stays closed."""
    seg = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg)])
    n_segments = max(int(np.ceil(cum_len[-1] / step)), 1)
    target = np.linspace(0, cum_len[-1], n_segments + 1)
    return np.column_stack([np.interp(target, cum_len, polyline[:, i]) for i in range(3)])


def curve_polyline(points, closed, step=CURVE_STEP_MM):
    """Exact spline through ordered points, sampled every `step` mm by arc length.
    For a closed curve the last point repeats the first one."""
    if closed:
        dense = fit_closed_spline(points, N_SPLINE_EVAL)
    else:
        dense = _fit_spline(points, per=0, n_dense=N_SPLINE_EVAL)
    return _resample_by_step(dense, step)


def point_to_polyline_distances(points, polyline):
    """Distance from each point to the nearest segment of the polyline (not its nearest vertex)."""
    p = np.asarray(points, dtype=float)[:, None, :]
    a = polyline[:-1][None, :, :]
    ab = (polyline[1:] - polyline[:-1])[None, :, :]
    length_sq = np.einsum('ijk,ijk->ij', ab, ab)
    t = np.einsum('ijk,ijk->ij', p - a, ab) / np.where(length_sq > 0, length_sq, 1.0)
    projection = a + np.clip(t, 0.0, 1.0)[..., None] * ab
    return np.linalg.norm(p - projection, axis=2).min(axis=1)


def _vertices(polyline, closed):
    return polyline[:-1] if closed else polyline


def point_to_curve_distance(gt_points, marked_points, closed):
    """Variant (a), MPCD: mean distance from the GT points as marked to the curve through the
    evaluated points."""
    polyline = curve_polyline(marked_points, closed)
    return float(np.mean(point_to_polyline_distances(gt_points, polyline)))


def symmetric_curve_distance(gt_points, marked_points, closed):
    """Variant (b), symmetric curve-to-curve distance. Both sides become splines sampled every
    CURVE_STEP_MM; the result is the half-sum of the two directed mean distances, so a longer
    curve does not outweigh a shorter one through its number of samples."""
    gt_line = curve_polyline(gt_points, closed)
    marked_line = curve_polyline(marked_points, closed)
    gt_to_marked = np.mean(point_to_polyline_distances(_vertices(gt_line, closed), marked_line))
    marked_to_gt = np.mean(point_to_polyline_distances(_vertices(marked_line, closed), gt_line))
    return float((gt_to_marked + marked_to_gt) / 2)
