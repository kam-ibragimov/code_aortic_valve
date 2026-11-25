"""Structures of every compared source (GT, network, observers) and the summary-table columns,
shared by the structural and the morphometric interobserver steps."""
import os
from pathlib import Path

import numpy as np

from data_preprocessing.oblique_slice_extractor import correct_br_points_to_plane
from data_preprocessing.text_worker import add_info_logging

from interobserver_study.slicer_txt_parser import get_case_number, parse_slicer_txt

LANDMARK_KEYS = ['R', 'L', 'N', 'RLC', 'RNC', 'LNC']
CURVE_KEYS = ['RGH', 'LGH', 'NGH', 'RCI', 'LCI', 'NCI']
PREDICTION_MARKER_KEY = 'RGH_pred'

DL_GROUPS = [('NL(1)', 'n'), ('PT(1)', 'p'), ('PT(2)', 'g')]


def _as_arrays(landmarks, curves):
    structures = {key: np.asarray(value, dtype=float) for key, value in landmarks.items()}
    structures.update({key: np.asarray(value, dtype=float) for key, value in curves.items()})
    return structures


def gt_structures(case):
    landmarks = {key: case[key][0] for key in LANDMARK_KEYS if key in case}
    curves = {key: case[key] for key in CURVE_KEYS if key in case}
    if 'BR - closed' in case:
        curves['BR'] = case['BR - closed']
    return _as_arrays(landmarks, curves)


def pred_structures(case):
    landmarks = {key: case[f'{key}_pred'] for key in LANDMARK_KEYS if f'{key}_pred' in case}
    curves = {key: case[f'{key}_pred'] for key in CURVE_KEYS + ['BR'] if f'{key}_pred' in case}
    return _as_arrays(landmarks, curves)


def observer_structures(parsed):
    landmarks = {key: parsed[key] for key in LANDMARK_KEYS if key in parsed}
    curves = {key: parsed[key] for key in CURVE_KEYS if key in parsed}
    if 'BR' in parsed:
        curves['BR'] = parsed['BR']
        if all(key in parsed for key in ('R', 'L', 'N')):
            _, _, curves['BR'] = correct_br_points_to_plane(parsed['BR'], r=parsed['R'],
                                                            l=parsed['L'], n=parsed['N'])
    return _as_arrays(landmarks, curves)


def load_observer_set(set_folder, dict_all_case):
    structures = {}
    for txt_path in sorted(Path(set_folder).glob('*.txt')):
        number = get_case_number(txt_path.name)
        case_name = f'n{number}'
        if number is None or case_name not in dict_all_case:
            add_info_logging(f'{set_folder}: {txt_path.name} not matched to a case, skipping.',
                             'work_logger')
            continue
        structures[case_name] = observer_structures(parse_slicer_txt(txt_path))
    return structures


def load_sources(dict_all_case, interobserver_txt_points_folder):
    """{source: {case: structures}} for the network and both observer sets."""
    return {
        'DL': {case_name: pred_structures(case) for case_name, case in dict_all_case.items()
               if PREDICTION_MARKER_KEY in case},
        'Intra-observer': load_observer_set(
            os.path.join(interobserver_txt_points_folder, '2 set'), dict_all_case),
        'Inter-observer': load_observer_set(
            os.path.join(interobserver_txt_points_folder, '1 set'), dict_all_case),
    }


def build_columns(sources, include_overlap_columns, include_median_column=False):
    """Summary columns as (label, source, cases, stat), plus the cases shared by the network and
    both observers. stat is 'mean' (mean ± std) or 'median' (median [Q1–Q3]).
    include_median_column: put a median column of all network cases first."""
    dl_cases = sorted(sources['DL'])
    intra_cases = sorted(sources['Intra-observer'])
    inter_cases = sorted(sources['Inter-observer'])
    overlap = sorted(set(dl_cases) & set(intra_cases) & set(inter_cases))

    def column(label, source, cases, stat='mean'):
        return f'{label} (n={len(cases)})', source, cases, stat

    columns = []
    if include_median_column:
        columns.append(column('DL NL+PT median', 'DL', dl_cases, stat='median'))
    columns.append(column('DL NL+PT', 'DL', dl_cases))
    columns += [column(f'DL {label}', 'DL', [c for c in dl_cases if c.startswith(prefix)])
                for label, prefix in DL_GROUPS]
    columns.append(column('Intra', 'Intra-observer', intra_cases))
    if include_overlap_columns:
        columns.append(column('Intra overlap', 'Intra-observer', overlap))
    columns.append(column('Inter', 'Inter-observer', inter_cases))
    if include_overlap_columns:
        columns.append(column('Inter overlap', 'Inter-observer', overlap))
        columns.append(column('DL overlap', 'DL', overlap))
    return columns, overlap
