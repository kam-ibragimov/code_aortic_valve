"""Обрезка маски корня аорты плоскостями через надиры и комиссуры. Общая для построения
GT-масок и для консолидации предсказанных масок."""
import numpy as np
import SimpleITK as sitk

MARGIN_MM = 2.0


def _inward_plane(points, inner_point):
    """Плоскость через три точки: точка на плоскости и единичная нормаль в сторону inner_point.
    None, если точки лежат на одной прямой или inner_point лежит в самой плоскости."""
    p1, p2, p3 = (np.asarray(p, dtype=float) for p in points)
    normal = np.cross(p2 - p1, p3 - p1)
    length = np.linalg.norm(normal)
    if length < 1e-6:
        return None
    normal = normal / length
    side = float((inner_point - p1) @ normal)
    if abs(side) < 1e-6:
        return None
    return p1, normal * np.sign(side)


def cut_mask_between_planes(mask_image, nadirs, commissures, margin_mm=MARGIN_MM):
    """
    Оставляет в маске только воксели между плоскостью надиров и плоскостью комиссур, каждая из
    которых сдвинута наружу на margin_mm. Сторона задаётся явно: внутренняя сторона плоскости
    надиров смотрит на центр комиссур и наоборот.
    Точки — в мировых координатах. Возвращает массив (z, y, x) или None, если плоскости
    построить нельзя.
    """
    nadirs = np.asarray(nadirs, dtype=float)
    commissures = np.asarray(commissures, dtype=float)
    planes = [_inward_plane(nadirs, commissures.mean(axis=0)),
              _inward_plane(commissures, nadirs.mean(axis=0))]
    if any(plane is None for plane in planes):
        return None

    mask = sitk.GetArrayFromImage(mask_image)
    zyx = np.argwhere(mask > 0)
    direction = np.array(mask_image.GetDirection()).reshape(3, 3)
    world = (zyx[:, ::-1] * np.array(mask_image.GetSpacing())) @ direction.T
    world += np.array(mask_image.GetOrigin())

    keep = np.ones(len(zyx), dtype=bool)
    for point, normal in planes:
        keep &= (world - point) @ normal >= -margin_mm
    outside = zyx[~keep]
    mask[outside[:, 0], outside[:, 1], outside[:, 2]] = 0
    return mask
