import os
from datetime import datetime

import pandas as pd

from data_preprocessing.text_worker import add_info_logging

from interobserver_study.boxplots import plot_morphometry_boxplots
from interobserver_study.morphometry_metrics import (COMMISSURES, CUSPS, commissural_angle_keys,
                                                     compute_case_metrics)
from interobserver_study.sources import build_columns, gt_structures, load_sources
from interobserver_study.summary_table import build_summary, plot_summary


def _per_cusp(suffix):
    return [f'{prefix}{suffix}' for prefix in CUSPS]


def _per_commissure(suffix):
    return [f'{prefix}{suffix}' for prefix in COMMISSURES]


def _build_rows():
    """Row order of the results table: (label, metric keys pooled in the row)."""
    length_groups = [('gH', _per_cusp('gH')), ('cI', _per_cusp('cI')), ('eFM', _per_cusp('eFM'))]
    height_groups = [('eH', _per_cusp('eH')), ('cH', _per_commissure('cH'))]
    angle_groups = [('bA', _per_cusp('bA')), ('cA', commissural_angle_keys())]

    def expand(groups):
        rows = []
        for name, keys in groups:
            rows.append((f'  {name}, all', keys))
            rows.extend((f'    {key}', [key]) for key in keys)
        return rows

    all_lengths = ['brP'] + [key for _, keys in length_groups for key in keys]
    all_heights = [key for _, keys in height_groups for key in keys]
    all_angles = [key for _, keys in angle_groups for key in keys]
    return ([('Lengths, all (mm)', all_lengths), ('  brP', ['brP'])] + expand(length_groups)
            + [('Heights, all (mm)', all_heights)] + expand(height_groups)
            + [('Angle, all (deg)', all_angles)] + expand(angle_groups))


def _belly_rows(source, case_name, segments, metrics):
    return [{'case': case_name, 'source': source, 'metric': key,
             'angle_deg': metrics[key],
             'curve_mm': segment['curve_mm'],
             'segment_start_mm': segment['start_mm'],
             'segment_end_mm': segment['end_mm'],
             'max_deviation_mm': segment['max_deviation_mm'],
             'used_fallback': segment['used_fallback']}
            for key, segment in segments.items()]


def process_interobserver_morphometry(result_folder, dict_all_case, interobserver_txt_points_folder,
                                      include_overlap_columns=True):
    """include_overlap_columns: add the columns limited to cases shared by the network and both
    observers (Intra, Inter and DL). The per-case CSV files always contain every case."""
    date_str = datetime.now().strftime('%d_%m_%y')
    output_folder = os.path.join(result_folder, f'Interobserver_morphometry_{date_str}')
    os.makedirs(output_folder, exist_ok=True)

    sources = load_sources(dict_all_case, interobserver_txt_points_folder)

    used_cases = set().union(*sources.values())
    belly_rows = []
    gt_metrics = {}
    for case_name in sorted(used_cases):
        gt_metrics[case_name], segments = compute_case_metrics(gt_structures(dict_all_case[case_name]))
        belly_rows.extend(_belly_rows('GT', case_name, segments, gt_metrics[case_name]))

    per_case_rows = []
    for source, cases in sources.items():
        for case_name, structures in cases.items():
            metrics, segments = compute_case_metrics(structures)
            belly_rows.extend(_belly_rows(source, case_name, segments, metrics))
            for key, value in metrics.items():
                gt_value = gt_metrics[case_name].get(key)
                if gt_value is None:
                    continue
                per_case_rows.append({'case': case_name, 'source': source, 'metric': key,
                                      'value': value, 'gt_value': gt_value,
                                      'abs_error': abs(value - gt_value)})
    per_case = pd.DataFrame(per_case_rows)

    columns, overlap = build_columns(sources, include_overlap_columns)
    summary = build_summary(per_case, _build_rows(), columns, value_col='abs_error')

    per_case.to_csv(os.path.join(output_folder, 'morphometry_per_case.csv'), index=False)
    pd.DataFrame(belly_rows).to_csv(os.path.join(output_folder, 'belly_segments.csv'), index=False)
    summary.to_csv(os.path.join(output_folder, 'morphometry_summary.csv'), index=False,
                   encoding='utf-8-sig')

    plot_summary(summary, os.path.join(output_folder, 'morphometry_summary.png'))
    plot_morphometry_boxplots(
        per_case, os.path.join(output_folder, 'boxplots_morphometry.png'),
        length_groups=[('brP', ['brP']), ('gH', _per_cusp('gH')), ('cI', _per_cusp('cI')),
                       ('eFM', _per_cusp('eFM')), ('eH', _per_cusp('eH')),
                       ('cH', _per_commissure('cH'))],
        angle_groups=[('bA', _per_cusp('bA')), ('cA', commissural_angle_keys())])

    add_info_logging(f'Interobserver morphometry completed: {len(sources["DL"])} DL cases, '
                     f'{len(overlap)} cases shared with observers ({", ".join(overlap)}).',
                     'result_logger')
