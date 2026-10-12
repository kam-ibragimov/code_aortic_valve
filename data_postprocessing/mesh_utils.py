# Vendored from MeshMetrics 0.2.1 (https://github.com/gasperpodobnik/MeshMetrics,
# commit 2a888b1686dfef49eff3a5dfdd0c7aefae545703, MeshMetrics/utils.py),
# Copyright Gasper Podobnik, Tomaz Vrtovec, licensed under the Apache License 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Code copied unchanged.
#
# Why vendored: vtk_analysis.CenterlineExtractor needs only these helpers, and
# MeshMetrics 1.4 (2026-10-11) renamed the package, changed np2sitk's signature
# and switched vtk_3D_meshing from SurfaceNets to discrete marching cubes, which
# shifts the clipped GH/CI curve ends. Keeping the validated version here makes
# the core install (and aortic-seg-infer's archive) independent of that package.

from pathlib import Path
from typing import Union

import numpy as np
import SimpleITK as sitk
import vtk

from SimpleITK.utilities.vtk import sitk2vtk


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
    elif isinstance(img, sitk.Image):
        pass
    else:
        raise NotImplementedError(f"Unknown image type: {type(img)}")
    return img


def vtk_2D_meshing(
    src_img: Union[str, Path, sitk.Image], pad: bool = True
) -> vtk.vtkPolyData:
    src_img = to_sitk(src_img)

    n_dim = src_img.GetDimension()
    assert n_dim == 2, "Only 2D images are supported for marching squares"

    if sitk.GetArrayFromImage(src_img).sum() == 0:
        return vtk.vtkPolyData()

    # pad to avoid potential open boundary related issues
    if pad:
        src_img = sitk.ConstantPad(src_img, (1, 1), (1, 1), 0)

    vtkImage = sitk2vtk(src_img > 0)
    vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
    meshing_alg = vtk.vtkSurfaceNets2D()
    meshing_alg.SmoothingOff()
    meshing_alg.SetInputData(vtkImage)
    meshing_alg.Update()
    mesh = meshing_alg.GetOutput()

    return mesh


def vtk_3D_meshing(
    src_img: Union[str, Path, sitk.Image], pad: bool = True
) -> vtk.vtkPolyData:
    src_img = to_sitk(src_img)

    n_dim = src_img.GetDimension()
    assert n_dim == 3, "Only 3D images are supported for marching cubes"

    if sitk.GetArrayViewFromImage(src_img).sum() == 0:
        return vtk.vtkPolyData()

    # pad to avoid potential open boundary related issues
    if pad:
        src_img = sitk.ConstantPad(src_img, (1, 1, 1), (1, 1, 1), 0)

    vtkImage = sitk2vtk(src_img > 0)
    vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
    meshing_alg = vtk.vtkSurfaceNets3D()
    meshing_alg.SmoothingOff()
    meshing_alg.SetOutputMeshTypeToTriangles()
    meshing_alg.SetInputData(vtkImage)
    meshing_alg.Update()
    mesh = meshing_alg.GetOutput()

    return mesh


def vtk_meshing(src_img: Union[str, Path, sitk.Image]):
    src_img = to_sitk(src_img)

    n_dim = src_img.GetDimension()
    if n_dim == 2:
        mesh = vtk_2D_meshing(src_img)
    elif n_dim == 3:
        mesh = vtk_3D_meshing(src_img)
    else:
        raise ValueError("Only 2D or 3D images are supported")
    return mesh
