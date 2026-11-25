"""Геометрическая консолидация маски корня аорты: предсказанная маска обрезается плоскостями
через предсказанные надиры и комиссуры, по тем же правилам, по которым построена GT-маска."""
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from data_postprocessing.mask_analysis import keep_largest_component
from data_preprocessing.mask_cutting import MARGIN_MM, cut_mask_between_planes
from data_preprocessing.text_worker import add_info_logging

NADIR_KEYS = ('R', 'L', 'N')
COMMISSURE_KEYS = ('RLC', 'RNC', 'LNC')


def _consolidate_case(source_path, output_path, case_points, margin_mm):
    """Обрезает одну маску и пишет её в output_path. Если обрезать нельзя, пишет исходную маску,
    чтобы набор кейсов в анализе не менялся. Возвращает строку отчёта."""
    image = sitk.ReadImage(str(source_path))
    voxels_before = int(np.count_nonzero(sitk.GetArrayViewFromImage(image)))

    cut, status = None, 'ok'
    missing = [key for key in NADIR_KEYS + COMMISSURE_KEYS if case_points.get(f'{key}_pred') is None]
    if missing:
        status = f'missing landmarks: {" ".join(missing)}'
    else:
        cut = cut_mask_between_planes(image,
                                      [case_points[f'{key}_pred'] for key in NADIR_KEYS],
                                      [case_points[f'{key}_pred'] for key in COMMISSURE_KEYS],
                                      margin_mm)
        if cut is None:
            status = 'degenerate planes'
        elif not cut.any():
            cut, status = None, 'empty after cutting'

    if cut is None:
        sitk.WriteImage(image, str(output_path), useCompression=True)
        voxels_cut, removed_components = voxels_before, 0
    else:
        new_image = sitk.GetImageFromArray(cut)
        new_image.CopyInformation(image)
        sitk.WriteImage(new_image, str(output_path), useCompression=True)
        voxels_cut = int(np.count_nonzero(cut))
        removed_components = keep_largest_component(output_path)

    return {'voxels_before': voxels_before,
            'voxels_removed_by_planes': voxels_before - voxels_cut,
            'voxels_removed_as_components': removed_components,
            'voxels_after': voxels_cut - removed_components,
            'status': status}


def consolidate_aorta_masks(source_folder, output_folder, dict_all_case, margin_mm=MARGIN_MM):
    """Консолидация всех масок папки source_folder (исходные файлы не меняются).
    Возвращает список строк отчёта, по одной на кейс."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    report = []
    for source_path in sorted(Path(source_folder).glob('*.nii.gz')):
        case_name = source_path.name[:-len('.nii.gz')]
        row = _consolidate_case(source_path, output_folder / source_path.name,
                                dict_all_case.get(case_name, {}), margin_mm)
        if row['status'] != 'ok':
            add_info_logging(f'Aorta mask consolidation: {case_name} not cut, original mask saved '
                             f'({row["status"]}).', 'work_logger')
        report.append({'case': case_name, **row})

    not_cut = [row['case'] for row in report if row['status'] != 'ok']
    add_info_logging(f'Aorta mask consolidation: {len(report) - len(not_cut)} of {len(report)} masks '
                     f'cut; not cut, original saved: {", ".join(not_cut) or "none"}.', 'result_logger')
    return report
