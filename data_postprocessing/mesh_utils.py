# Mesh helpers for vtk_analysis.CenterlineExtractor: the surface of a binary mask, used to clip
# the GH/CI centre-line curves at the mask boundary. The algorithm chosen here sets where the
# clipped curve ends, so changing it shifts the GH/CI curve ends.
#
# Own code, no MeshMetrics dependency (2026-11-27). History:
#   - 2025-12-11 .. 2026-10-12: vtk_meshing came from MeshMetrics (git install), which used
#     SurfaceNets since 2025-08-20; 2026-10-12 that SurfaceNets version (0.2.1, 2a888b1) was
#     vendored here.
#   - The GH/CI curves in dict_all_case and the published results (run of 2026-10-12) were,
#     however, built with discrete marching cubes - the algorithm MeshMetrics returned to on
#     2026-10-03 (v1.0+). Marching cubes reproduces those curves (264/336 within 0.05 mm,
#     SurfaceNets 27/336) and lies closer to the GT curves (MPCD GH 0.694 vs 0.711 mm,
#     CI 0.542 vs 0.567 mm), so it is used here.
# The mask is meshed in index space and the points are then mapped to world coordinates
# (spacing, origin, direction), so no SimpleITK<->VTK image conversion package is needed.

from pathlib import Path
from typing import Union

import numpy as np
import SimpleITK as sitk
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy


def np2sitk(img_np: np.ndarray, spacing, swapaxes=True) -> sitk.Image:
    if swapaxes:
        assert img_np.ndim in [2, 3], "Unsupported number of dimensions"
        img_np = np.swapaxes(img_np, 0, -1)
    img_sitk = sitk.GetImageFromArray(img_np)
    img_sitk.SetSpacing(spacing)
    return img_sitk


def to_sitk(img: Union[str, Path, sitk.Image]) -> sitk.Image:
    if isinstance(img, (str, Path)):
        img = sitk.ReadImage(str(img))
    elif not isinstance(img, sitk.Image):
        raise NotImplementedError(f"Unknown image type: {type(img)}")
    return img


def vtk_meshing(src_img: Union[str, Path, sitk.Image], pad: bool = True) -> vtk.vtkPolyData:
    """Surface of the foreground (> 0) of a 2D/3D mask in world coordinates: discrete marching
    cubes in 3D, discrete flying edges (a contour in the z=0 plane) in 2D.
    pad: one background voxel around the image, so a mask touching the border stays closed."""
    src_img = to_sitk(src_img)
    n_dim = src_img.GetDimension()
    if n_dim not in (2, 3):
        raise ValueError("Only 2D or 3D images are supported")
    if not sitk.GetArrayViewFromImage(src_img).any():
        return vtk.vtkPolyData()
    if pad:
        src_img = sitk.ConstantPad(src_img, (1,) * n_dim, (1,) * n_dim, 0)

    mask = (sitk.GetArrayViewFromImage(src_img) > 0).astype(np.uint8)
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(*src_img.GetSize(), *(1,) * (3 - n_dim))
    vtk_img.GetPointData().SetScalars(numpy_to_vtk(mask.ravel(), deep=True))

    if n_dim == 3:
        meshing_alg = vtk.vtkDiscreteMarchingCubes()
        meshing_alg.ComputeNormalsOff()
        meshing_alg.ComputeGradientsOff()
    else:
        meshing_alg = vtk.vtkDiscreteFlyingEdges2D()
    meshing_alg.SetValue(0, 1)
    meshing_alg.ComputeScalarsOff()
    meshing_alg.SetInputData(vtk_img)
    meshing_alg.Update()
    mesh = meshing_alg.GetOutput()
    if mesh.GetNumberOfPoints() == 0:
        return vtk.vtkPolyData()

    # index -> world coordinates
    index = vtk_to_numpy(mesh.GetPoints().GetData())[:, :n_dim]
    direction = np.array(src_img.GetDirection()).reshape(n_dim, n_dim)
    world = (index * np.array(src_img.GetSpacing())) @ direction.T + np.array(src_img.GetOrigin())
    if n_dim == 2:
        world = np.column_stack([world, np.zeros(len(world))])
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(world, dtype=float), deep=True))
    mesh.SetPoints(points)
    return mesh
