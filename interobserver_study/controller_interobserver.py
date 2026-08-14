import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data_preprocessing.text_worker import add_info_logging
from data_postprocessing.plotting_graphs import plot_table
from data_preprocessing.oblique_slice_extractor import correct_br_points_to_plane

from interobserver_study.slicer_txt_parser import parse_slicer_txt, get_case_number
from interobserver_study.curve_metrics import open_curve_mpcd, closed_curve_mpcd

LANDMARK_ORDER = ['R', 'L', 'N', 'RLC', 'RNC', 'LNC']
GH_KEYS = ['RGH', 'LGH', 'NGH']
CI_KEYS = ['RCI', 'LCI', 'NCI']

STRUCTURE_GROUPS = [
    ('Landmarks', LANDMARK_ORDER),
    ('Geometric Height (GH)', GH_KEYS),
    ('Cusp Insertion (CI)', CI_KEYS),
    ('Basal Ring (BR)', ['BR']),
]


def _landmark_errors(parsed, gt_case):
    row = {}
    for key in LANDMARK_ORDER:
        if key in parsed and key in gt_case:
            row[key] = float(np.linalg.norm(np.array(parsed[key]) - np.array(gt_case[key][0])))
    return row


def _curve_errors(parsed, gt_case, keys):
    row = {}
    for key in keys:
        if key in parsed and key in gt_case:
            row[key] = open_curve_mpcd(parsed[key], gt_case[key])
    return row


def _br_error(parsed, gt_case):
    if 'BR' not in parsed or 'BR - closed' not in gt_case:
        return None
    r, l, n = parsed.get('R'), parsed.get('L'), parsed.get('N')
    br_points = parsed['BR']
    if r is not None and l is not None and n is not None:
        _, _, br_points = correct_br_points_to_plane(br_points, r=r, l=l, n=n)
    return closed_curve_mpcd(br_points, gt_case['BR - closed'])


def _process_set(set_folder, dict_all_case, has_br):
    rows = []

    for txt_path in sorted(Path(set_folder).glob('*.txt')):
        num = get_case_number(txt_path.name)
        if num is None:
            continue
        case_name = f'n{num}'
        gt_case = dict_all_case.get(case_name)
        if gt_case is None:
            add_info_logging(f'{set_folder}: {case_name} not found in dict_all_case, skipping.',
                             'work_logger')
            continue

        parsed = parse_slicer_txt(txt_path)

        row = {'case': case_name}
        row.update(_landmark_errors(parsed, gt_case))
        row.update(_curve_errors(parsed, gt_case, GH_KEYS))
        row.update(_curve_errors(parsed, gt_case, CI_KEYS))
        if has_br:
            br_mpcd = _br_error(parsed, gt_case)
            if br_mpcd is not None:
                row['BR'] = br_mpcd

        rows.append(row)

    return pd.DataFrame(rows)


def _build_summary_table(df):
    data_table = []
    for label, keys in STRUCTURE_GROUPS:
        present_keys = [key for key in keys if key in df.columns]
        pooled = df[present_keys].to_numpy().ravel() if present_keys else np.array([])
        pooled = pooled[~pd.isna(pooled)]
        n_cases = int(df[present_keys].notna().any(axis=1).sum()) if present_keys else 0
        if len(pooled):
            data_table.append([label, round(float(pooled.mean()), 2), round(float(pooled.std()), 2), n_cases])
        else:
            data_table.append([label, '—', '—', 0])
    return data_table


def process_interobserver_comparison(result_folder, dict_all_case, interobserver_txt_points_folder):
    date_str = datetime.now().strftime('%d_%m_%y')
    base_folder = os.path.join(result_folder, f'Interobserver_study_{date_str}')

    set_configs = [
        (os.path.join(interobserver_txt_points_folder, '1 set'), '1_set_interobserver', False),
        (os.path.join(interobserver_txt_points_folder, '2 set'), '2_set_intraobserver', True),
    ]

    for set_folder, set_dirname, has_br in set_configs:
        df = _process_set(set_folder, dict_all_case, has_br)
        set_base = os.path.join(base_folder, set_dirname)
        os.makedirs(set_base, exist_ok=True)

        df.to_csv(os.path.join(set_base, f'{set_dirname}.csv'), index=False)

        data_table = _build_summary_table(df)
        columns = ['Structure', 'Mean, mm', 'Std, mm', 'N cases']
        plot_table(data_table, columns, os.path.join(set_base, f'{set_dirname}.png'))

    add_info_logging('Interobserver curve/landmark comparison completed', 'result_logger')