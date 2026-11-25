import math

import numpy as np

from data_postprocessing.curve_utils import resample_curve
from interobserver_study.curve_metrics import _fit_spline, fit_closed_spline

# name prefix -> nadir, GH curve, CI curve, the two commissures of the cusp
CUSPS = {
    'r': ('R', 'RGH', 'RCI', ('RLC', 'RNC')),
    'l': ('L', 'LGH', 'LCI', ('RLC', 'LNC')),
    'n': ('N', 'NGH', 'NCI', ('RNC', 'LNC')),
}
# name prefix -> commissure; with the 'cH' suffix this gives rlcH, rncH, lncH
COMMISSURES = {'rl': 'RLC', 'rn': 'RNC', 'ln': 'LNC'}

BELLY_STEP_MM = 0.5
BELLY_TOLERANCE_MM = 0.5
BELLY_START_FRACTION = 1 / 3
BELLY_MIN_LENGTH_MM = 4.0
BELLY_FALLBACK_WINDOW = (0.1, 0.6)

N_DENSE = 300


def commissural_angle_keys():
    return [f'{prefix}cA' for prefix in CUSPS]


def _drop_repeated_points(points):
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return pts
    keep = np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-6
    return pts[np.concatenate([[True], keep])]


def dense_open_curve(points):
    """Exact spline through ordered points, used for every source so lengths stay comparable."""
    pts = _drop_repeated_points(points)
    if len(pts) < 2:
        return None
    return _fit_spline(pts, per=0, n_dense=N_DENSE)


def dense_closed_curve(points):
    pts = _drop_repeated_points(points)
    if len(pts) < 3:
        return None
    return fit_closed_spline(pts, N_DENSE)


def polyline_length(pts):
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def _unit_normal(a, b, c):
    normal = np.cross(np.asarray(b) - np.asarray(a), np.asarray(c) - np.asarray(a))
    norm = np.linalg.norm(normal)
    return None if norm < 1e-9 else normal / norm


def _orient_from_nadir(dense, nadir):
    start_dist = np.linalg.norm(dense[0] - nadir)
    end_dist = np.linalg.norm(dense[-1] - nadir)
    return dense if start_dist <= end_dist else dense[::-1]


def _fit_line(pts):
    """PCA line through pts: unit direction and the largest perpendicular deviation."""
    centered = pts - pts.mean(axis=0)
    direction = np.linalg.svd(centered, full_matrices=False)[2][0]
    residuals = centered - np.outer(centered @ direction, direction)
    return direction, float(np.linalg.norm(residuals, axis=1).max())


def find_belly_segment(gh_from_nadir):
    """Longest nearly straight segment that starts in the first part of the GH curve.

    The doctor draws the line through the main straight part near the nadir: the very start
    may be skipped, and a bend may be missing, weak or followed by a very short tail, so the
    segment is searched for directly rather than derived from a bend.
    """
    total = polyline_length(gh_from_nadir)
    n_points = max(2, round(total / BELLY_STEP_MM) + 1)
    pts = np.array(resample_curve(gh_from_nadir, n_points))
    step = total / (n_points - 1)
    max_start = int(BELLY_START_FRACTION * (n_points - 1))
    min_points = math.ceil(BELLY_MIN_LENGTH_MM / step) + 1

    for size in range(n_points, min_points - 1, -1):
        best = None
        for start in range(min(max_start, n_points - size) + 1):
            direction, deviation = _fit_line(pts[start:start + size])
            if deviation <= BELLY_TOLERANCE_MM and (best is None or deviation < best[2]):
                best = (start, direction, deviation)
        if best is not None:
            start, direction, deviation = best
            return {'direction': direction, 'start_mm': start * step,
                    'end_mm': (start + size - 1) * step, 'curve_mm': total,
                    'max_deviation_mm': deviation, 'used_fallback': False}

    start = round(BELLY_FALLBACK_WINDOW[0] * (n_points - 1))
    end = max(start + 1, round(BELLY_FALLBACK_WINDOW[1] * (n_points - 1)))
    direction, deviation = _fit_line(pts[start:end + 1])
    return {'direction': direction, 'start_mm': start * step, 'end_mm': end * step,
            'curve_mm': total, 'max_deviation_mm': deviation, 'used_fallback': True}


def _commissural_angles(structures, tips, commissural_normal):
    """Angle between the commissural plane and the plane C1, C2, Q of each cusp (Q is the GH tip)."""
    angles = {}
    for prefix, (_, _, _, (c1_key, c2_key)) in CUSPS.items():
        if prefix in tips and c1_key in structures and c2_key in structures:
            cusp_normal = _unit_normal(structures[c1_key], structures[c2_key], tips[prefix])
            if cusp_normal is not None:
                cosine = min(1.0, abs(float(commissural_normal @ cusp_normal)))
                angles[f'{prefix}cA'] = math.degrees(math.acos(cosine))
    return angles


def compute_case_metrics(structures):
    """Morphometric values of one case from one source.

    structures: landmarks and ordered curve points as arrays, keys R, L, N, RLC, RNC, LNC,
    RGH, LGH, NGH, RCI, LCI, NCI, BR. A missing structure only drops the metrics using it.

    Returns (metrics, belly_segments), both keyed by metric name.
    """
    metrics = {}
    belly_segments = {}

    if 'BR' in structures:
        dense = dense_closed_curve(structures['BR'])
        if dense is not None:
            metrics['brP'] = polyline_length(dense)

    br_normal = None
    if all(key in structures for key in ('R', 'L', 'N')):
        br_normal = _unit_normal(structures['R'], structures['L'], structures['N'])

    commissural_normal = None
    if all(key in structures for key in COMMISSURES.values()):
        commissural_normal = _unit_normal(*(structures[key] for key in COMMISSURES.values()))

    tips = {}
    for prefix, (nadir_key, gh_key, ci_key, (c1_key, c2_key)) in CUSPS.items():
        if ci_key in structures:
            dense = dense_open_curve(structures[ci_key])
            if dense is not None:
                metrics[f'{prefix}cI'] = polyline_length(dense)

        if gh_key not in structures or nadir_key not in structures:
            continue
        dense = dense_open_curve(structures[gh_key])
        if dense is None:
            continue
        gh_curve = _orient_from_nadir(dense, structures[nadir_key])
        tip = gh_curve[-1]
        tips[prefix] = tip
        metrics[f'{prefix}gH'] = polyline_length(gh_curve)

        if c1_key in structures and c2_key in structures:
            metrics[f'{prefix}eFM'] = float(np.linalg.norm(structures[c1_key] - tip)
                                            + np.linalg.norm(structures[c2_key] - tip))

        if br_normal is not None:
            metrics[f'{prefix}eH'] = abs(float((tip - structures['R']) @ br_normal))
            segment = find_belly_segment(gh_curve)
            sine = min(1.0, abs(float(segment['direction'] @ br_normal)))
            metrics[f'{prefix}bA'] = math.degrees(math.asin(sine))
            belly_segments[f'{prefix}bA'] = segment

    if br_normal is not None:
        for prefix, key in COMMISSURES.items():
            if key in structures:
                metrics[f'{prefix}cH'] = abs(float((structures[key] - structures['R']) @ br_normal))

    if commissural_normal is not None:
        metrics.update(_commissural_angles(structures, tips, commissural_normal))

    return metrics, belly_segments
