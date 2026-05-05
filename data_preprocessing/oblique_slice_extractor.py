import numpy as np
import SimpleITK as sitk
from pathlib import Path


def get_oblique_plane_params(r, l, n):
    """
    Computes oblique plane parameters from 3 nadir points R, L, N.

    x_axis = direction L→R (normalized)
    normal = cross(L-R, N-R), normalized, forced to positive Z component
    y_axis = cross(normal, x_axis)

    Returns:
        centroid : np.ndarray, center of R/L/N in world coords (mm)
        x_axis   : np.ndarray, unit vector (in-plane, along L-R)
        y_axis   : np.ndarray, unit vector (in-plane, perpendicular)
        normal   : np.ndarray, unit normal to the plane
    """
    r = np.array(r, dtype=float)
    l = np.array(l, dtype=float)
    n = np.array(n, dtype=float)

    centroid = (r + l + n) / 3.0

    x_axis = l - r
    x_axis = x_axis / np.linalg.norm(x_axis)

    normal = np.cross(l - r, n - r)
    normal = normal / np.linalg.norm(normal)

    # Consistent direction: force positive Z component
    if normal[2] < 0:
        normal = -normal

    y_axis = np.cross(normal, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    return centroid, x_axis, y_axis, normal


def _build_resampler(centroid, x_axis, y_axis, normal, spacing, size_hw, fill_value, interpolator):
    """
    Creates a configured SimpleITK ResampleImageFilter for an oblique plane.

    The output image has size [W, H, 1], centered at centroid.
    Direction matrix columns: [x_axis | y_axis | normal].
    """
    H, W = size_hw

    # Origin = world coord of voxel [0, 0, 0]
    origin = (np.array(centroid)
              - (W / 2.0) * spacing * x_axis
              - (H / 2.0) * spacing * y_axis)

    # Direction stored row-major: each row is one world axis (X, Y, Z)
    # Columns represent image axes i, j, k → x_axis, y_axis, normal
    direction = [x_axis[0], y_axis[0], normal[0],
                 x_axis[1], y_axis[1], normal[1],
                 x_axis[2], y_axis[2], normal[2]]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([W, H, 1])
    resampler.SetOutputSpacing([float(spacing), float(spacing), float(spacing)])
    resampler.SetOutputOrigin(origin.tolist())
    resampler.SetOutputDirection(direction)
    resampler.SetDefaultPixelValue(float(fill_value))
    resampler.SetInterpolator(interpolator)

    return resampler


def find_global_2d_size(mask_crop_folder, dict_all_case, initial_size=512, padding=10):
    """
    Finds the global 2D output size needed to contain the aorta mask
    for every case when projected onto its oblique valve plane.

    Strategy:
      - For each case, project the cropped aorta mask onto the oblique plane
        using a large initial canvas centred at the centroid of R/L/N.
      - Measure how far the non-zero region extends from the centre (in pixels)
        in both directions for each axis.
      - Take the maximum half-extent across all cases, then build a symmetric
        canvas: H = 2 * max_h_half + 2*padding, W = 2 * max_w_half + 2*padding.

    Returns:
        [H, W] in pixels (both values are even numbers)
    """
    max_h_half = 0
    max_w_half = 0
    center = initial_size / 2.0

    for case_name, points_dict in dict_all_case.items():
        mask_path = Path(mask_crop_folder) / f"{case_name}.nii.gz"
        if not mask_path.exists():
            continue

        r = points_dict['R'][0]
        l = points_dict['L'][0]
        n = points_dict['N'][0]

        centroid, x_axis, y_axis, normal = get_oblique_plane_params(r, l, n)

        image = sitk.ReadImage(str(mask_path))
        spacing = image.GetSpacing()[0]  # isotropic

        resampler = _build_resampler(
            centroid, x_axis, y_axis, normal,
            spacing, (initial_size, initial_size),
            fill_value=0,
            interpolator=sitk.sitkNearestNeighbor
        )
        result_array = sitk.GetArrayFromImage(resampler.Execute(image))[0]  # (H, W)

        nonzero = np.argwhere(result_array > 0)
        if len(nonzero) == 0:
            continue

        h_min, w_min = nonzero.min(axis=0)
        h_max, w_max = nonzero.max(axis=0)

        # How far the mask extends from the centre on each side
        h_half = max(center - h_min, h_max - center)
        w_half = max(center - w_min, w_max - center)

        max_h_half = max(max_h_half, h_half)
        max_w_half = max(max_w_half, w_half)

    # Build symmetric canvas, round up to even number
    H = int(np.ceil((2 * max_h_half + 2 * padding) / 2)) * 2
    W = int(np.ceil((2 * max_w_half + 2 * padding) / 2)) * 2

    return [H, W]


def extract_2d_slice_pair(image_3d_path, image_2d_path,
                          r, l, n, size_hw,
                          mask_3d_path=None, mask_2d_path=None):
    """
    Extracts oblique 2D slices for one case: CT image and optionally aorta mask.

    Spacing is read from the CT image (image and mask share the same spacing).
    The plane is defined by nadir points R, L, N in world coordinates.

    Saved as NIfTI [W, H, 1] — ready for nnUNet 2D.

    Parameters:
        image_3d_path : path to cropped 3D CT image
        image_2d_path : output path for 2D CT slice
        r, l, n       : nadir points in world coordinates (each a list/array of 3 floats)
        size_hw       : [H, W] output size in pixels
        mask_3d_path  : path to cropped 3D aorta mask (optional)
        mask_2d_path  : output path for 2D mask slice (optional)
    """
    image = sitk.ReadImage(str(image_3d_path))
    spacing = image.GetSpacing()[0]  # isotropic

    centroid, x_axis, y_axis, normal = get_oblique_plane_params(r, l, n)

    ct_resampler = _build_resampler(
        centroid, x_axis, y_axis, normal,
        spacing, size_hw, fill_value=-1000,
        interpolator=sitk.sitkLinear
    )
    sitk.WriteImage(ct_resampler.Execute(image), str(image_2d_path))

    if mask_3d_path is not None and mask_2d_path is not None:
        mask = sitk.ReadImage(str(mask_3d_path))
        mask_resampler = _build_resampler(
            centroid, x_axis, y_axis, normal,
            spacing, size_hw, fill_value=0,
            interpolator=sitk.sitkNearestNeighbor
        )
        sitk.WriteImage(mask_resampler.Execute(mask), str(mask_2d_path))


def compute_br_plane_offset(br_points, r, l, n, t_threshold=2.5, min_offset_mm=0.3):
    """
    Computes the signed offset of 'BR - closed' points from the oblique plane R/L/N.

    For each BR point computes signed distance: dot(point - centroid, normal).
    Detection uses a t-statistic: |mean| / (std / sqrt(N)) > t_threshold.
    This is robust to annotation noise regardless of the number of points.

    t_threshold=2.5 corresponds roughly to p<0.02 for typical N (10-20 points).
    min_offset_mm guards against flagging noise when std itself is very small.

    Returns the mean signed offset (mm) when systematic, 0.0 otherwise.
    """
    centroid, _, _, normal = get_oblique_plane_params(r, l, n)
    pts = np.array(br_points, dtype=float)
    signed_dists = (pts - centroid) @ normal

    n = len(signed_dists)
    mean_d = float(np.mean(signed_dists))
    std_d = float(np.std(signed_dists))

    if std_d < 1e-6:
        return round(mean_d, 3) if abs(mean_d) > min_offset_mm else 0.0

    t_stat = abs(mean_d) / (std_d / np.sqrt(n))
    if t_stat > t_threshold and abs(mean_d) > min_offset_mm:
        return round(mean_d, 3)
    return 0.0