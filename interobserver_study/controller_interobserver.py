import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data_preprocessing.text_worker import add_info_logging

from interobserver_study.boxplots import plot_geometry_boxplots
from interobserver_study.case_ranking import build_case_ranking
from interobserver_study.curve_metrics import point_to_curve_distance, symmetric_curve_distance
from interobserver_study.sources import (CURVE_KEYS, LANDMARK_KEYS, build_columns, gt_structures,
                                         load_sources)
from interobserver_study.summary_table import build_summary, plot_summary

NADIRS = ['R', 'L', 'N']
COMMISSURES = ['RLC', 'RNC', 'LNC']
GH_KEYS = ['RGH', 'LGH', 'NGH']
CI_KEYS = ['RCI', 'LCI', 'NCI']
DSC_KEY = 'AorticRoot'

# Two ways to measure curves:
# (a) MPCD — GT points to the evaluated curve; (b) CCD — symmetric curve-to-curve distance.
CURVE_MEASURES = {
    'mpcd': ('MPCD', point_to_curve_distance),
    'symmetric': ('CCD', symmetric_curve_distance),
}

OBSERVER_SET_DIRS = {'Inter-observer': '1_set_interobserver',
                     'Intra-observer': '2_set_intraobserver'}


def _build_rows(curve_measure):
    """Row order of the results table: (label, metric keys pooled in the row)."""
    def group(name, keys):
        return [(f'  {name}, all', keys)] + [(f'    {key}', [key]) for key in keys]

    return ([('Aortic root (DSC)', [DSC_KEY])]
            + [('Landmarks, all (PPD, mm)', NADIRS + COMMISSURES)]
            + group('Nadirs', NADIRS) + group('Commissures', COMMISSURES)
            + [(f'Curves, all ({curve_measure}, mm)', ['BR'] + GH_KEYS + CI_KEYS), ('  BR', ['BR'])]
            + group('GH', GH_KEYS) + group('CI', CI_KEYS))


def _latest_result_folder(result_folder, ds_folder_name, file_name):
    """Latest result/<ds_folder_name>_dd_mm_yy folder holding file_name. Folders with any other
    suffix (manual copies, experiments) are ignored."""
    pattern = re.compile(rf'^{re.escape(ds_folder_name)}_(\d{{2}}_\d{{2}}_\d{{2}})$')
    candidates = []
    for folder in Path(result_folder).iterdir():
        match = pattern.match(folder.name)
        if folder.is_dir() and match and (folder / file_name).is_file():
            candidates.append((datetime.strptime(match.group(1), '%d_%m_%y'), folder))
    return max(candidates)[1] if candidates else None


def _dsc_rows(result_folder, aorta_segment_ds_name, dl_cases):
    folder = _latest_result_folder(result_folder, aorta_segment_ds_name, 'per_case_metrics.csv')
    if folder is None:
        add_info_logging(f'No per_case_metrics.csv for {aorta_segment_ds_name}: DSC will be N/A.',
                         'result_logger')
        return []
    df = pd.read_csv(folder / 'per_case_metrics.csv')
    add_info_logging(f'Interobserver structures: DSC taken from {folder.name}', 'result_logger')
    df = df[df['case'].isin(dl_cases)]
    return [{'case': case, 'source': 'DL', 'metric': DSC_KEY, 'measure': measure, 'value': value}
            for measure, column in (('DSC', 'Dice'), ('HD', 'HD'))
            for case, value in zip(df['case'], df[column])]


def _case_rows(source, case_name, structures, gt):
    rows = []

    def add(metric, measure, value):
        rows.append({'case': case_name, 'source': source, 'metric': metric,
                     'measure': measure, 'value': value})

    for key in LANDMARK_KEYS:
        if key in structures and key in gt:
            add(key, 'PPD', float(np.linalg.norm(structures[key] - gt[key])))
    for key in CURVE_KEYS + ['BR']:
        if key in structures and key in gt:
            for measure, distance in CURVE_MEASURES.values():
                add(key, measure, distance(gt[key], structures[key], closed=(key == 'BR')))
    return rows


def _save_observer_set_csv(per_case, base_folder, source, set_dirname):
    """Per-set wide CSV (one row per case), as before: point distances and MPCD of the curves."""
    subset = per_case[(per_case['source'] == source) & per_case['measure'].isin(['PPD', 'MPCD'])]
    wide = subset.pivot(index='case', columns='metric', values='value')
    wide = wide.reindex(columns=[key for key in LANDMARK_KEYS + GH_KEYS + CI_KEYS + ['BR']
                                 if key in wide.columns]).reset_index()
    set_folder = os.path.join(base_folder, set_dirname)
    os.makedirs(set_folder, exist_ok=True)
    wide.to_csv(os.path.join(set_folder, f'{set_dirname}.csv'), index=False)


def process_interobserver_comparison(result_folder, dict_all_case, interobserver_txt_points_folder,
                                     aorta_segment_ds_name, include_overlap_columns=False,
                                     include_median_column=True):
    """Structural accuracy of the network and both observers against GT: DSC of the aortic root
    (network only), point distances of the landmarks and two curve distances (two tables).
    include_overlap_columns: add the columns limited to cases shared by the network and both
    observers; few cases, for a look only.
    include_median_column: add a median [Q1–Q3] column of all network cases before the mean."""
    date_str = datetime.now().strftime('%d_%m_%y')
    base_folder = os.path.join(result_folder, f'Interobserver_study_{date_str}')
    os.makedirs(base_folder, exist_ok=True)

    sources = load_sources(dict_all_case, interobserver_txt_points_folder)

    rows = _dsc_rows(result_folder, aorta_segment_ds_name, set(sources['DL']))
    for source, cases in sources.items():
        for case_name, structures in sorted(cases.items()):
            rows.extend(_case_rows(source, case_name, structures,
                                   gt_structures(dict_all_case[case_name])))
    per_case = pd.DataFrame(rows)
    per_case.to_csv(os.path.join(base_folder, 'structures_per_case.csv'), index=False)

    for source, set_dirname in OBSERVER_SET_DIRS.items():
        _save_observer_set_csv(per_case, base_folder, source, set_dirname)

    plot_geometry_boxplots(per_case, os.path.join(base_folder, 'boxplots_geometry.png'),
                           landmark_keys=LANDMARK_KEYS,
                           curve_groups=[('GH', GH_KEYS), ('CI', CI_KEYS), ('BR', ['BR'])],
                           dsc_key=DSC_KEY)
    build_case_ranking(per_case).to_csv(os.path.join(base_folder, 'cases_quality_ranking.csv'),
                                        index=False)

    columns, overlap = build_columns(sources, include_overlap_columns, include_median_column)
    for suffix, (measure, _) in CURVE_MEASURES.items():
        subset = per_case[per_case['measure'].isin(['DSC', 'PPD', measure])]
        summary = build_summary(subset, _build_rows(measure), columns, value_col='value')
        summary.to_csv(os.path.join(base_folder, f'structures_summary_{suffix}.csv'), index=False,
                       encoding='utf-8-sig')
        plot_summary(summary, os.path.join(base_folder, f'structures_summary_{suffix}.png'))

    add_info_logging(f'Interobserver structures completed: {len(sources["DL"])} DL cases, '
                     f'{len(overlap)} cases shared with observers.', 'result_logger')
