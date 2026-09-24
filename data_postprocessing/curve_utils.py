import numpy as np
import SimpleITK as sitk


def resample_curve(pts, n_points):
    diffs = np.diff(pts, axis=0)
    cum_len = np.concatenate([[0], np.cumsum(np.linalg.norm(diffs, axis=1))])
    target = np.linspace(0, cum_len[-1], n_points)
    resampled = np.column_stack([np.interp(target, cum_len, pts[:, i]) for i in range(3)])
    return resampled.tolist()


def sample_closed_curve_uniform(points, n_points):
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


def load_labels_mask_sitk(file_path, label):
    mask_img = sitk.ReadImage(file_path)
    masks_array = sitk.GetArrayFromImage(mask_img)
    mask_binary = (masks_array == label).astype(np.uint8)

    mask_sitk = sitk.GetImageFromArray(mask_binary)
    mask_sitk.SetOrigin(mask_img.GetOrigin())
    mask_sitk.SetSpacing(mask_img.GetSpacing())
    mask_sitk.SetDirection(mask_img.GetDirection())

    return mask_sitk


def vtk_to_numpy(vtk_curve):
    return np.array([vtk_curve.GetPoints().GetPoint(i)
                     for i in range(vtk_curve.GetNumberOfPoints())])


def apply_bezier_anchor(pts, anchor_pt, blend_fraction):
    """Replace the nearer endpoint tail with a cubic Bezier that lands exactly on anchor_pt.

    Auto-detects which end (first or last point) is closer to the anchor,
    flips the array if needed so the logic always works on the tail,
    then flips back before returning.
    """
    anchor = np.array(anchor_pt, dtype=float)

    dist0 = np.linalg.norm(pts[0] - anchor)
    distn = np.linalg.norm(pts[-1] - anchor)
    flip = dist0 < distn
    if flip:
        pts = pts[::-1].copy()

    n = len(pts)
    blend_idx = max(1, int(n * (1.0 - blend_fraction)))

    # Stable tangent: average direction over a small window before blend_idx
    window = max(3, int(n * 0.03))
    tangent = pts[blend_idx] - pts[max(0, blend_idx - window)]
    t_norm = np.linalg.norm(tangent)
    if t_norm < 1e-9:
        tangent = pts[-1] - pts[0]
        t_norm = np.linalg.norm(tangent)
    tangent = tangent / t_norm

    P0, P3 = pts[blend_idx], anchor
    dist = np.linalg.norm(P3 - P0)

    # P1 continues the existing curve direction; P2 approaches anchor naturally
    P1 = P0 + tangent * dist * 0.4
    approach = P0 - P3
    a_norm = np.linalg.norm(approach)
    P2 = P3 + (approach / a_norm) * dist * 0.4 if a_norm > 1e-9 else P3

    n_bez = max(20, n - blend_idx)
    t = np.linspace(0, 1, n_bez)
    bezier = (
        ((1 - t) ** 3)[:, None] * P0
        + 3 * ((1 - t) ** 2 * t)[:, None] * P1
        + 3 * ((1 - t) * t ** 2)[:, None] * P2
        + (t ** 3)[:, None] * P3
    )

    modified = np.vstack([pts[:blend_idx], bezier])
    return modified[::-1].copy() if flip else modified
