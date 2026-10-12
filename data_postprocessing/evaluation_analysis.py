import numpy as np
import SimpleITK as sitk
from sklearn.metrics import jaccard_score, f1_score
from typing import Literal, Tuple
from medpy.metric.binary import hd, assd


def compute_metrics_gh_line(coord_org, coord_pred):
    def _mean_euclidean_distance(A, B):
        return np.mean(np.linalg.norm(A - B, axis=1))

    def _rmsd(A, B):
        A, B = np.asarray(A), np.asarray(B)
        return np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1)))

    def _std_distance(A, B):
        """
        Стандартное отклонение расстояний между парами точек.
        A, B — массивы размером (N, 3)
        """
        assert A.shape == B.shape, "Наборы точек должны иметь одинаковую форму"
        distances = np.linalg.norm(A - B, axis=1)
        return np.std(distances)

    def _polyline_length(A):
        """Вычисление длины ломаной по соседним точкам."""
        diffs = np.diff(A, axis=0)
        return np.sum(np.linalg.norm(diffs, axis=1))

    def _length_difference(A, B):
        """Разница длин двух ломаных линий."""
        len_A = _polyline_length(A)
        len_B = _polyline_length(B)
        return abs(len_A - len_B)

    return {
        "RMSD": _rmsd(coord_org, coord_pred),
        "MED": _mean_euclidean_distance(coord_org, coord_pred),
        "SD": _std_distance(coord_org, coord_pred),
        "LengthDiff": _length_difference(coord_org, coord_pred)
    }


def compute_metrics_gh_line_v2(coord_org, vtk_curve_pred):
    pts = np.array(vtk_curve_pred.GetPoints().GetData())
    distances = [np.sqrt(np.sum((pts - p) ** 2, axis=1)).min() for p in coord_org]
    return {"MPCD": np.mean(distances)}


class landmarking_locked:
    """пока не используется"""

    def __init__(self, probability_maps):
        self.probability_maps = probability_maps

    def __prepare_probability_lists(self):
        # Step 1: Extract nonzero probability locations for each landmark
        nonzero_locs = {}  # Store voxel locations where probability > 0
        probs = {}  # Store corresponding probabilities

        for key, prob_map in self.probability_maps.items():
            indices = np.argwhere(prob_map > 0)  # Get all non-zero locations
            probabilities = prob_map[prob_map > 0]  # Get probability values

            # Normalize probabilities to sum to 1 (so they form a proper distribution)
            probabilities /= np.sum(probabilities)

            # Store valid locations and their probabilities
            nonzero_locs[key] = indices
            probs[key] = probabilities

        return nonzero_locs, probs

    def __measure_metrics(self, landmarks):
        pass

    def simulation_MC(self, num_simulations):
        # Step 2: Monte Carlo Sampling
        results = []  # Store weighted measurements

        nonzero_locs, probs = self.__prepare_probability_lists()

        for _ in range(num_simulations):
            sampled_landmarks = {}

            # Step 3: Randomly select a voxel location for each landmark
            for key in self.probability_maps.keys():
                sampled_index = np.random.choice(len(probs[key]), p=probs[key])  # Weighted random selection
                sampled_landmarks[key] = nonzero_locs[key][sampled_index]  # Get the selected voxel

            # Step 4: Compute measurement (Replace with actual function)
            measurement = self.__measure_metrics(sampled_landmarks)

            # Step 5: Store weighted measurement (measurement * probability product)
            weight = np.prod([probs[key][sampled_index] for key in self.probability_maps.keys()])
            results.append(measurement * weight)

        return results


def _crop_to_union_bbox(a: np.ndarray, b: np.ndarray, margin: int = 2):
    """
    Вырезает из обеих масок один общий bounding box ненулевых вокселей (объединение a и b)
    с запасом margin, не выходя за границы массива. Оба массива режутся одними срезами,
    поэтому взаимное расположение масок сохраняется. Вне бокса обе маски нулевые, так что
    Dice/IoU/HD/ASSD не меняются, а distance transform считается на малом объёме.
    """
    union = (a > 0) | (b > 0)
    if not union.any():
        return a, b
    slices = []
    for axis in range(union.ndim):
        other = tuple(i for i in range(union.ndim) if i != axis)
        idx = np.flatnonzero(union.any(axis=other))
        slices.append(slice(max(idx[0] - margin, 0), min(idx[-1] + margin + 1, union.shape[axis])))
    slices = tuple(slices)
    a_crop, b_crop = a[slices], b[slices]
    if (np.count_nonzero(a_crop) != np.count_nonzero(a) or
            np.count_nonzero(b_crop) != np.count_nonzero(b)):
        raise RuntimeError("Bounding-box crop lost foreground voxels")
    return a_crop, b_crop


def evaluate_segmentation(true_mask: np.ndarray, pred_mask: np.ndarray, num_classes: int = 1,
                          average: Literal['macro', 'weighted'] = 'macro'):
    """
    Вычисляет Dice и IoU между масками.

    Parameters:
        true_mask (np.ndarray): Ground truth маска (2D или 3D).
        pred_mask (np.ndarray): Предсказанная маска (2D или 3D).
        num_classes (int): Количество классов. Если 1 — бинарная сегментация.
        average (str): Способ усреднения ('macro' или 'weighted').

    Returns:
        dict: {'Dice': ..., 'IoU': ...}
    """
    metrics = {}

    if num_classes == 1:
        true_mask, pred_mask = _crop_to_union_bbox(true_mask, pred_mask)

        # То же, что sklearn f1_score / jaccard_score (pos_label=1, zero_division -> 0.0),
        # но без flatten-копий и float-массивов на весь объём
        true_pos = true_mask == 1
        pred_pos = pred_mask == 1
        tp = int(np.count_nonzero(true_pos & pred_pos))
        n_true = int(np.count_nonzero(true_pos))
        n_pred = int(np.count_nonzero(pred_pos))
        del true_pos, pred_pos
        union = n_true + n_pred - tp

        metrics["Dice"] = 2 * tp / (n_true + n_pred) if (n_true + n_pred) > 0 else 0.0
        metrics["IoU"] = tp / union if union > 0 else 0.0

        try:
            # Преобразуем в бинарные маски
            true_bin = (true_mask > 0).astype(np.bool_)
            pred_bin = (pred_mask > 0).astype(np.bool_)

            if np.count_nonzero(true_bin) == 0 or np.count_nonzero(pred_bin) == 0:
                metrics["HD"] = np.nan
                metrics["ASSD"] = np.nan
            else:
                metrics["HD"] = hd(pred_bin, true_bin)
                metrics["ASSD"] = assd(pred_bin, true_bin)
        except Exception as e:
            metrics["HD"] = np.nan
            metrics["ASSD"] = np.nan
    else:
        true_flat = true_mask.flatten()
        pred_flat = pred_mask.flatten()
        dice = f1_score(true_flat, pred_flat, average=average, labels=range(num_classes))
        iou = jaccard_score(true_flat, pred_flat, average=average, labels=range(num_classes))
        metrics["Dice"] = dice
        metrics["IoU"] = iou
        metrics["HD"] = np.nan  # многоклассовую HD/ASSD сложнее интерпретировать
        metrics["ASSD"] = np.nan

    return metrics


def hausdorff_distance_sitk(mask1: np.ndarray, mask2: np.ndarray,
                            spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
    """
    Вычисляет Hausdorff и HD95 расстояния между двумя 3D масками с учетом физического размера вокселя.

    Parameters:
        mask1 (np.ndarray): Ground truth бинарная 3D маска.
        mask2 (np.ndarray): Предсказанная бинарная 3D маска.
        spacing (tuple): Физический размер вокселя (мм), по осям (z, y, x).

    Returns:
        dict: {'Hausdorff': ..., 'HD95': ...}
    """
    mask1_itk = sitk.GetImageFromArray(mask1.astype(np.uint8))
    mask2_itk = sitk.GetImageFromArray(mask2.astype(np.uint8))

    mask1_itk.SetSpacing(spacing)
    mask2_itk.SetSpacing(spacing)

    hd_filter = sitk.HausdorffDistanceImageFilter()
    hd_filter.Execute(mask1_itk, mask2_itk)

    return {
        'Hausdorff': hd_filter.GetHausdorffDistance(),
        'HD95': hd_filter.Get95PercentHausdorffDistance()
    }

