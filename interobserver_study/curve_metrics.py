import numpy as np
from scipy.interpolate import splprep, splev

from data_postprocessing.curve_utils import (_resample_curve, _sample_closed_curve_uniform,
                                             mean_point_to_curve_distance)


def _fit_spline(points, per, n_dense):
    pts = np.asarray(points, dtype=float)
    k = min(3, len(pts) - 1)
    tck, _ = splprep(pts.T, s=0, k=k, per=per)
    u = np.linspace(0, 1, n_dense)
    return np.array(splev(u, tck)).T


def open_curve_mpcd(marked_points, gt_points, n_curve_points=100, n_dense=300):
    """Fits an exact open cubic spline through ordered marked points (GH/CI),
    resamples it by arc length, and returns the MPCD against gt_points."""
    dense = _fit_spline(marked_points, per=0, n_dense=n_dense)
    resampled = _resample_curve(dense, n_curve_points)
    return mean_point_to_curve_distance(gt_points, resampled)


def closed_curve_mpcd(marked_points, gt_points, n_curve_points=100, n_dense=300):
    """Fits an exact closed (periodic) cubic spline through ordered marked
    points (BR), resamples it by arc length, and returns the MPCD against
    gt_points."""
    dense = _fit_spline(marked_points, per=1, n_dense=n_dense)
    resampled = _sample_closed_curve_uniform(dense, n_curve_points)
    return mean_point_to_curve_distance(gt_points, resampled)