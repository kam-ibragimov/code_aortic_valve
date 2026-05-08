import os
import json
from ftplib import all_errors

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from datetime import datetime
import SimpleITK as sitk
from typing import Dict, List
from data_postprocessing.evaluation_analysis import compute_metrics_gh_line, compute_metrics_gh_line_v2
from data_postprocessing.montecarlo import LandmarkingMonteCarlo
from data_postprocessing.mask_analysis import (mask_comparison, LandmarkCentersCalculator,
                                               new_spline_from_pixel_coord, load_new_coords_org,
                                               extract_boundary_curve_world)
from data_postprocessing.plotting_graphs import summarize_and_plot, plot_group_comparison, plot_table
from data_preprocessing.text_worker import add_info_logging
from models.controller_nnUnet import process_nnunet
from data_postprocessing.metrics_config import metric_to_landmarks
from data_postprocessing.geometric_metrics import controller_metrics
from data_postprocessing.vtk_analysis import CenterlineExtractor


def load_mask(file_path, probabilities_map=True):
    if probabilities_map:
        mask_array = np.load(file_path)
        labels = len(mask_array["probabilities"])
        return mask_array, labels
    else:
        mask_img = sitk.ReadImage(file_path)
        mask_array = sitk.GetArrayFromImage(mask_img)
        labels = int(np.unique(mask_array).max()) + 1
        return mask_array, labels


def load_labels_mask_sitk(file_path, label):
    mask_img = sitk.ReadImage(file_path)
    masks_array = sitk.GetArrayFromImage(mask_img)
    mask_binary = (masks_array == label).astype(np.uint8)

    mask_sitk = sitk.GetImageFromArray(mask_binary)
    mask_sitk.SetOrigin(mask_img.GetOrigin())
    mask_sitk.SetSpacing(mask_img.GetSpacing())
    mask_sitk.SetDirection(mask_img.GetDirection())

    return mask_sitk


def _sample_curve_points(vtk_curve, n_points):
    pts = np.array([[vtk_curve.GetPoints().GetPoint(i)[j] for j in range(3)]
                    for i in range(vtk_curve.GetNumberOfPoints())])
    if len(pts) < 2:
        return None
    diffs = np.diff(pts, axis=0)
    cum_len = np.concatenate([[0], np.cumsum(np.linalg.norm(diffs, axis=1))])
    target = np.linspace(0, cum_len[-1], n_points)
    resampled = np.column_stack([np.interp(target, cum_len, pts[:, i]) for i in range(3)])
    return resampled.tolist()


def curve_lines_analysis(data_path,
                         result_path,
                         folder_name,
                         dict_cases,
                         keys_to_need,
                         name_result_folder,
                         spline_smoothing=1.0,
                         probabilities_map=True,
                         original_mask=False,
                         points2points=False,
                         curve2points=False,
                         n_curve_points=None):

    date_str = datetime.now().strftime("%d_%m_%y")
    result_folder_path = os.path.join(result_path, f"{folder_name}_{date_str}")
    os.makedirs(result_folder_path, exist_ok=True)
    predictions = {}
    nnUNet_folder = os.path.join(data_path, "nnUNet_folder")
    result_mask_folder = os.path.join(nnUNet_folder, "nnUNet_test", folder_name)
    original_mask_folder = os.path.join(nnUNet_folder, "original_mask", folder_name)

    group_label_map = {
        "all": "All",
        "g": "Ger. path.",
        "p": "Slo. path.",
        "n": "Slo. norm."
    }

    if points2points:
        per_case_p2p_csv = os.path.join(str(result_folder_path), "per_case_p2p_metrics.csv")
        metrics_name = ["RMSD", "MED", "SD", "LengthDiff"]

        if os.path.exists(per_case_p2p_csv):
            # Загружаем данные из файлов
            add_info_logging("Using cached metrics from CSV files", "work_logger")
            df = pd.read_csv(per_case_p2p_csv)
        else:
            per_case_p2p_data = []

            ext = "*.npz" if probabilities_map else "*.nii.gz"
            files = list(Path(result_mask_folder).glob(ext))

            for file_path in files:
                rmsd_metric = []
                med_metric = []
                sd_metric = []
                ld_metric = []
                metrics = {}
                case_name = file_path.name[:-4] if probabilities_map else file_path.name[:-7]
                first_char = case_name[0]
                masks_pred, levels_pred = load_mask(file_path, probabilities_map)
                original_mask_path = os.path.join(original_mask_folder, case_name + ".nii.gz")
                for label in range(1, levels_pred):
                    if probabilities_map:
                        mask_pred = masks_pred["probabilities"][label]
                    else:
                        mask_pred = (masks_pred == label).astype(np.uint8)
                    new_coords_org = load_new_coords_org(mask_path=original_mask_path,
                                                         label=label,
                                                         coord_org=dict_cases[case_name][keys_to_need[label]],
                                                         original_mask=original_mask)
                    new_coords_pred = new_spline_from_pixel_coord(mask_pred, str(original_mask_path))
                    temp_metrics = compute_metrics_gh_line(new_coords_org, new_coords_pred)
                    rmsd_metric.append(temp_metrics["RMSD"])
                    med_metric.append(temp_metrics["MED"])
                    sd_metric.append(temp_metrics["SD"])
                    ld_metric.append(temp_metrics["LengthDiff"])
                metrics["case"] = case_name
                metrics["group"] = first_char
                metrics["RMSD"] = np.mean(rmsd_metric)
                metrics["MED"] = np.mean(med_metric)
                metrics["SD"] = np.mean(sd_metric)
                metrics["LengthDiff"] = np.mean(ld_metric)
                per_case_p2p_data.append(metrics)

            # Сохраняем метрики по кейсам
            df = pd.DataFrame(per_case_p2p_data)
            df.to_csv(per_case_p2p_csv, index=False)

        for metric_name in metrics_name:
            data_for_plot = df[['group', metric_name]].dropna(how='any')
            plot_group_comparison('group', metric_name, group_label_map, data_for_plot,
                                  os.path.join(str(result_folder_path), name_result_folder))
        data_table = [
            ["All",
             round(df["RMSD"].mean(numeric_only=True), 2),
             round(df["MED"].mean(numeric_only=True), 2),
             round(df["SD"].mean(numeric_only=True), 2),
             round(df["LengthDiff"].mean(numeric_only=True), 2),
             int(len(df["group"]))],
            ["German\npathology",
             round(df[df['group'] == "g"]["RMSD"].mean(numeric_only=True), 2),
             round(df[df['group'] == "g"]["MED"].mean(numeric_only=True), 2),
             round(df[df['group'] == "g"]["SD"].mean(numeric_only=True), 2),
             round(df[df['group'] == "g"]["LengthDiff"].mean(numeric_only=True), 2),
             int(len(df[df['group'] == "g"]))],
            ["Slovenian\npathology",
             round(df[df['group'] == "p"]["RMSD"].mean(numeric_only=True), 2),
             round(df[df['group'] == "p"]["MED"].mean(numeric_only=True), 2),
             round(df[df['group'] == "p"]["SD"].mean(numeric_only=True), 2),
             round(df[df['group'] == "p"]["LengthDiff"].mean(numeric_only=True), 2),
             int(len(df[df['group'] == "p"]))],
            ["Slovenian\nnormal",
             round(df[df['group'] == "n"]["RMSD"].mean(numeric_only=True), 2),
             round(df[df['group'] == "n"]["MED"].mean(numeric_only=True), 2),
             round(df[df['group'] == "n"]["SD"].mean(numeric_only=True), 2),
             round(df[df['group'] == "n"]["LengthDiff"].mean(numeric_only=True), 2),
             int(len(df[df['group'] == "n"]))],
        ]
        columns = ["Type", "Root Mean Square\nDistancer, mm", "Mean Euclidean\nDistance, mm",
                   "Standard Deviation\nof Distances, mm", "Mean Length\nDifference, mm", "Number of\nimages"]

        results_table_p2p_path = os.path.join(result_folder_path, f"landmark_errors_p2p_{folder_name}.png")
        plot_table(data_table, columns, results_table_p2p_path)

    if curve2points:
        per_case_c2p_csv = os.path.join(str(result_folder_path), "per_case_c2p_metrics.csv")
        not_found_csv = os.path.join(str(result_folder_path), "not_found_curves.csv")
        predictions_json_path = os.path.join(str(result_folder_path), "curve_predictions.json") if n_curve_points else None
        metrics_name_2 = ["MPCD"]

        use_cache = os.path.exists(per_case_c2p_csv) and (
            n_curve_points is None or (predictions_json_path and os.path.exists(predictions_json_path))
        )

        if use_cache:
            add_info_logging("Using cached metrics from CSV files", "work_logger")
            df_2 = pd.read_csv(per_case_c2p_csv)
            df_nf = pd.read_csv(not_found_csv) if os.path.exists(not_found_csv) else pd.DataFrame()
            if predictions_json_path and os.path.exists(predictions_json_path):
                with open(predictions_json_path) as f:
                    predictions = json.load(f)
        else:
            per_case_c2p_data = []
            not_found_data = []

            ext = "*.npz" if probabilities_map else "*.nii.gz"
            files = list(Path(result_mask_folder).glob(ext))
            vtk_curve_extractor = CenterlineExtractor(spline_smoothing=spline_smoothing)

            for file_path in files:
                asd_metric = []
                metrics_2 = {}
                case_name = file_path.name[:-4] if probabilities_map else file_path.name[:-7]
                first_char = case_name[0]
                masks_pred, levels_pred = load_mask(file_path, probabilities_map)
                for label in range(1, levels_pred):
                    if probabilities_map:
                        mask_pred = masks_pred["probabilities"][label]
                    else:
                        mask_pred = (masks_pred == label).astype(np.uint8)
                    mask_sitk = load_labels_mask_sitk(file_path, label)
                    vtk_curve_pred = vtk_curve_extractor.run(mask_pred, mask_sitk)
                    if vtk_curve_pred.GetNumberOfPoints() == 0:
                        add_info_logging(
                            f"Curve not found: case={case_name}, label={keys_to_need[label]}",
                            "work_logger"
                        )
                        not_found_data.append({
                            "case": case_name,
                            "group": first_char,
                            "label": keys_to_need[label]
                        })
                        asd_metric.append(np.nan)
                        continue
                    temp_metrics_2 = compute_metrics_gh_line_v2(dict_cases[case_name][keys_to_need[label]], vtk_curve_pred)
                    asd_metric.append(temp_metrics_2["MPCD"])
                    if n_curve_points:
                        sampled = _sample_curve_points(vtk_curve_pred, n_curve_points)
                        if sampled is not None:
                            if case_name not in predictions:
                                predictions[case_name] = {}
                            predictions[case_name][keys_to_need[label] + "_pred"] = sampled
                metrics_2["case"] = case_name
                metrics_2["group"] = first_char
                asd_arr = np.array(asd_metric, dtype=float)
                metrics_2["MPCD"] = float(np.nanmean(asd_arr)) if not np.all(np.isnan(asd_arr)) else np.nan
                per_case_c2p_data.append(metrics_2)

            df_2 = pd.DataFrame(per_case_c2p_data)
            df_2.to_csv(per_case_c2p_csv, index=False)

            df_nf = pd.DataFrame(not_found_data)
            if not df_nf.empty:
                df_nf.to_csv(not_found_csv, index=False)

            if predictions_json_path:
                with open(predictions_json_path, "w") as f:
                    json.dump(predictions, f, indent=2)

        for metric_name in metrics_name_2:
            data_for_plot = df_2[['group', metric_name]].dropna(how='any')
            plot_group_comparison('group', metric_name, group_label_map, data_for_plot,
                                  os.path.join(str(result_folder_path), name_result_folder, "_2"))
        data_table_2 = [
            ["All",
             round(df_2["MPCD"].mean(numeric_only=True), 2),
             int(len(df_2["group"]))],
            ["German\npathology",
             round(df_2[df_2['group'] == "g"]["MPCD"].mean(numeric_only=True), 2),
             int(len(df_2[df_2['group'] == "g"]))],
            ["Slovenian\npathology",
             round(df_2[df_2['group'] == "p"]["MPCD"].mean(numeric_only=True), 2),
             int(len(df_2[df_2['group'] == "p"]))],
            ["Slovenian\nnormal",
             round(df_2[df_2['group'] == "n"]["MPCD"].mean(numeric_only=True), 2),
             int(len(df_2[df_2['group'] == "n"]))],
        ]
        columns_2 = ["Type", "Mean Point-to-Curve\nDistance, mm", "Number of\nimages"]
        results_table_c2p_path = os.path.join(result_folder_path, f"landmark_errors_c2p_{folder_name}.png")
        plot_table(data_table_2, columns_2, results_table_c2p_path)

        if not df_nf.empty:
            label_names = list(keys_to_need.values())
            groups = ["g", "p", "n"]
            group_display = {"g": "German\npathology", "p": "Slovenian\npathology", "n": "Slovenian\nnormal"}
            total_cases = len(df_2)
            nf_table = [["All"] + [int((df_nf["label"] == lbl).sum()) for lbl in label_names] + [total_cases]]
            for grp in groups:
                grp_total = int((df_2["group"] == grp).sum())
                nf_table.append(
                    [group_display[grp]] +
                    [int(((df_nf["label"] == lbl) & (df_nf["group"] == grp)).sum()) for lbl in label_names] +
                    [grp_total]
                )
            nf_columns = ["Type"] + [f"Not found\n{lbl}" for lbl in label_names] + ["Total\nimages"]
            nf_table_path = os.path.join(result_folder_path, f"not_found_curves_{folder_name}.png")
            plot_table(nf_table, nf_columns, nf_table_path)
            add_info_logging(f"Not found curves total: {len(df_nf)}", "work_logger")

    if points2points or curve2points:
        add_info_logging("Analysis completed", "work_logger")

    return predictions


def _sample_closed_curve_uniform(points, n_points):
    """Sample n_points equally spaced (by arc length) on a closed curve.
    The wrap-around segment (last→first) is included so all N gaps are equal.
    Returns list of [x, y, z] — first and last points are distinct.
    """
    pts = np.asarray(points, dtype=float)
    pts_closed = np.vstack([pts, pts[0:1]])          # close the loop for arc-length calc
    seg_lengths = np.linalg.norm(np.diff(pts_closed, axis=0), axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cum_len[-1]
    if total_length == 0 or len(pts) < 2:
        return None
    targets = np.arange(n_points) * (total_length / n_points)
    resampled = np.column_stack([np.interp(targets, cum_len, pts_closed[:, i]) for i in range(3)])
    return resampled.tolist()


def _detect_plane_offset(distances, cv_threshold=0.20, min_mean_mm=0.5):
    """True if all GT-to-curve distances are approximately equal, indicating
    a systematic shift (e.g. wrong annotation plane).
    Criterion: CV = std/mean < cv_threshold.
    min_mean_mm guards against flagging near-perfect predictions."""
    mean_d = float(np.mean(distances))
    if mean_d < min_mean_mm:
        return False
    return float(np.std(distances)) / mean_d < cv_threshold


def basal_ring_analysis(data_path, result_path, folder_name, dict_cases,
                        plane_offset_mode="none", n_curve_points=None):
    # plane_offset_mode: "none"   — все смещения False
    #                    "detect" — статистическое вычисление _detect_plane_offset
    #                    "key"    — читать "BR - plane offset" из dict_cases
    date_str = datetime.now().strftime("%d_%m_%y")
    result_folder_path = os.path.join(result_path, f"{folder_name}_{date_str}")
    os.makedirs(result_folder_path, exist_ok=True)

    nnUNet_folder = os.path.join(data_path, "nnUNet_folder")
    result_mask_folder = os.path.join(nnUNet_folder, "nnUNet_test", folder_name)

    per_case_csv = os.path.join(str(result_folder_path), "per_case_metrics_br.csv")
    predictions_json_path = os.path.join(str(result_folder_path), "br_predictions.json") if n_curve_points else None
    predictions = {}

    group_label_map = {
        "all": "All",
        "g": "Ger. path.",
        "p": "Slo. path.",
        "n": "Slo. norm."
    }

    needs_recompute = not os.path.exists(per_case_csv)
    if not needs_recompute:
        df = pd.read_csv(per_case_csv)
        if "plane_offset" not in df.columns:
            needs_recompute = True
    if not needs_recompute and predictions_json_path and not os.path.exists(predictions_json_path):
        needs_recompute = True

    if needs_recompute:
        per_case_data = []
        files = list(Path(result_mask_folder).glob("*.nii.gz"))

        for file_path in files:
            case_name = file_path.name[:-7]
            first_char = case_name[0]

            if case_name not in dict_cases or "BR - closed" not in dict_cases[case_name]:
                add_info_logging(f"Missing BR - closed for case: {case_name}", "work_logger")
                continue

            mask_sitk = sitk.ReadImage(str(file_path))
            mask_array = sitk.GetArrayFromImage(mask_sitk)

            pred_boundary = extract_boundary_curve_world(mask_array, mask_sitk)
            if len(pred_boundary) < 2:
                add_info_logging(f"Empty predicted boundary for case: {case_name}", "work_logger")
                continue

            gt_points = np.array(dict_cases[case_name]["BR - closed"])

            distances = [np.sqrt(np.sum((pred_boundary - p) ** 2, axis=1)).min() for p in gt_points]
            mpcd = float(np.mean(distances))
            if plane_offset_mode == "detect":
                offset = _detect_plane_offset(distances)
            elif plane_offset_mode == "key":
                offset = dict_cases[case_name].get("BR - plane offset", 0.0) != 0.0
            else:
                offset = False

            per_case_data.append({
                "case": case_name,
                "group": first_char,
                "MPCD": round(mpcd, 3),
                "plane_offset": offset
            })

            if n_curve_points:
                sampled = _sample_closed_curve_uniform(pred_boundary, n_curve_points)
                if sampled is not None:
                    predictions[case_name] = {"BR_pred": sampled}

        df = pd.DataFrame(per_case_data)
        df.to_csv(per_case_csv, index=False)
        if predictions_json_path:
            with open(predictions_json_path, "w") as f:
                json.dump(predictions, f, indent=2)
    else:
        add_info_logging("Using cached metrics from CSV files", "work_logger")
        if predictions_json_path and os.path.exists(predictions_json_path):
            with open(predictions_json_path) as f:
                predictions = json.load(f)

    if plane_offset_mode == "key":
        df["plane_offset"] = df["case"].map(
            lambda c: dict_cases.get(c, {}).get("BR - plane offset", 0.0) != 0.0
        )
    elif plane_offset_mode == "none":
        df["plane_offset"] = False

    df_valid = df[~df["plane_offset"]]

    data_for_plot = df_valid[['group', 'MPCD']].dropna(how='any')
    plot_group_comparison('group', 'MPCD', group_label_map, data_for_plot,
                          os.path.join(str(result_folder_path), "basal_ring_comparison"))

    def _group_stats(group_key):
        sub = df if group_key is None else df[df['group'] == group_key]
        total_all = len(sub)
        if total_all == 0:
            return float('nan'), 0, "0%"
        sub_valid = sub[~sub["plane_offset"]]
        total_valid = len(sub_valid)
        asd_mean = round(float(sub_valid["MPCD"].mean()), 2) if total_valid > 0 else float('nan')
        offset_pct = f"{round(100 * sub['plane_offset'].sum() / total_all, 1)}%"
        return asd_mean, total_valid, offset_pct

    groups = [
        ("All", None),
        ("German\npathology", "g"),
        ("Slovenian\npathology", "p"),
        ("Slovenian\nnormal", "n"),
    ]

    data_table = []
    for label, group_key in groups:
        asd_mean, total, offset_pct = _group_stats(group_key)
        data_table.append([label, asd_mean, total, offset_pct])

    columns = ["Type", "Mean Point-to-Curve\nDistance, mm", "Number of\nimages", "Plane offset\ncases, %"]
    plot_table(data_table, columns, os.path.join(result_folder_path, f"br_errors_{folder_name}.png"))
    add_info_logging("Basal ring analysis completed", "work_logger")
    return predictions


def mask_analysis(data_path, result_path, type_mask, folder_name):
    date_str = datetime.now().strftime("%d_%m_%y")
    result_folder_path = os.path.join(result_path, f"{folder_name}_{date_str}")
    os.makedirs(result_folder_path, exist_ok=True)
    per_case_csv = os.path.join(str(result_folder_path), "per_case_metrics.csv")

    group_label_map = {
        "all": "All",
        "g": "Ger. path.",
        "p": "Slo. path.",
        "n": "Slo. norm."
    }
    metrics = ["Dice", "IoU", "HD", "ASSD"]

    if os.path.exists(per_case_csv):
        # Загружаем данные из файлов
        add_info_logging("Using cached metrics from CSV files", "work_logger")
        df = pd.read_csv(per_case_csv)
    else:
        # Пересчитываем метрики
        per_case_data = mask_comparison(data_path, type_mask=type_mask, folder_name=folder_name)

        # Сохраняем метрики по кейсам
        df = pd.DataFrame(per_case_data)
        df.to_csv(per_case_csv, index=False)

    for metric_name in metrics:
        data_for_plot = df[['group', metric_name]].dropna(how='any')
        assd_mean = df["ASSD"].mean(numeric_only=True)
        assd_median = df["ASSD"].median(numeric_only=True)
        assd_std = df["ASSD"].std(numeric_only=True)
        dice_mean = df["Dice"].mean(numeric_only=True)
        dice_median = df["Dice"].median(numeric_only=True)
        dice_std = df["Dice"].std(numeric_only=True)
        assd_mean_groupby = df.groupby("group")["ASSD"].mean(numeric_only=True)
        assd_median_groupby = df.groupby("group")["ASSD"].median(numeric_only=True)
        assd_std_groupby = df.groupby("group")["ASSD"].std(numeric_only=True)
        dice_mean_groupby = df.groupby("group")["Dice"].mean(numeric_only=True)
        dice_median_groupby = df.groupby("group")["Dice"].median(numeric_only=True)
        dice_std_groupby = df.groupby("group")["Dice"].std(numeric_only=True)
        plot_group_comparison('group', metric_name, group_label_map, data_for_plot,
                              os.path.join(str(result_folder_path), "aorta_root_comparison"))
    add_info_logging("Analysis completed", "work_logger")


def experiment_analysis(data_path,
                        dict_case,
                        generate_result=False,
                        find_center_mass=False,
                        find_monte_carlo=False):
    nnUNet_folder = os.path.join(data_path, "nnUNet_folder")
    list_result_type = ["binary_map", "probability_map"]
    for radius, case_name in dict_case.items():
        ds_folder_name = f"Dataset{case_name}_AortaLandmarks"
        for type_map in list_result_type:
            if type_map == "probability_map":
                save_probabilities = True
            else:
                save_probabilities = False
            if generate_result:
                process_nnunet(folder=nnUNet_folder, ds_folder_name=ds_folder_name,
                               id_case=case_name, folder_image_path=None, folder_mask_path=None, dict_dataset={},
                               predicting_mod=True, save_probabilities=save_probabilities)
                add_info_logging(f"radius: {radius}, type predicting: {type_map}", "result_logger")
            landmarks_analysis(data_path, ds_folder_name,
                             find_center_mass=find_center_mass,
                             find_monte_carlo=find_monte_carlo,
                             probabilities_map=save_probabilities)
            # data_path = Path(data_path)
            # result_landmarks_folder = data_path / "nnUNet_folder" / "nnUNet_test" / ds_folder_name
            # original_mask_folder = data_path / "nnUNet_folder" / "original_mask" / ds_folder_name
            # json_path = data_path / "nnUNet_folder" / "json_info"


def landmarks_analysis(data_path, dict_all_case,
                       ds_folder_name,
                       find_center_mass=False,
                       find_monte_carlo=False,
                       probabilities_map=False,
                       type_set="six_landmarks",
                       active_point_name=None,
                       case_subset=None,
                       group_label="",
                       result_folder_name=None):

    def process_file(file, original_mask_folder, probabilities_map):
        res_test = LandmarkCentersCalculator()
        file_name = file.name[:-4] + ".nii.gz"
        if probabilities_map:
            pred = res_test.extract_landmarks_com_npz(original_mask_folder / file_name, file)
        else:
            pred = res_test.extract_landmarks_com_nii(file)
        return pred

    def _compute_errors(true, pred, results, file_name, threshold=3.6):
        for number_key, name_key in point_name.items():
            dist = None
            if number_key == 'all':
                continue
            if number_key in pred:
                dist = np.linalg.norm(np.array(true[name_key.upper()][0]) - pred[number_key])  # Евклидово расстояние
                if dist > threshold:
                    dist = None

            results.append({
                "filename": file_name,
                "group": file_name[0],
                "point_id": name_key,
                "error": dist
            })
        return results

    # add_info_logging("start analysis", "work_logger")
    data_path = Path(data_path)
    result_landmarks_folder = data_path / "nnUNet_folder" / "nnUNet_test" / ds_folder_name
    original_mask_folder = data_path / "nnUNet_folder" / "original_mask" / ds_folder_name
    json_path = data_path / "nnUNet_folder" / "json_info"
    date_str = datetime.now().strftime("%d_%m_%y")
    _result_name = result_folder_name if result_folder_name is not None else ds_folder_name
    result_folder_path = data_path / "result" / f"{_result_name}_{date_str}{group_label}"
    result_folder_path.mkdir(parents=True, exist_ok=True)
    results_csv_path = result_folder_path / f"landmark_errors_{ds_folder_name}.csv"

    if type_set == "six_landmarks":
        point_name = {"all": "All", 1:"r", 2:"l", 3:"n", 4:"rlc", 5:"rnc", 6:"lnc"}
    elif type_set == "gh_landmark":
        point_name = {1: "gh"}
    if active_point_name is not None:
        point_name = active_point_name
    type_label = {
        "all": "All",
        "g": "Ger. path.",
        "p": "Slo. path.",
        "n": "Slo. norm."
    }

    results = []  # список словарей
    predictions = {}  # {case_name: {"R_pred": [x,y,z], ...}}

    predictions_json_path = result_folder_path / f"landmark_predictions_{ds_folder_name}.json"

    if find_center_mass:
        if os.path.exists(results_csv_path) and os.path.exists(predictions_json_path):
            # Загружаем данные из файлов
            add_info_logging("Using cached metrics from CSV files", "work_logger")
            results_df = pd.read_csv(results_csv_path)
            with open(predictions_json_path, "r") as f:
                predictions = json.load(f)
        else:
            ext = "*.npz" if probabilities_map else "*.nii.gz"
            files = list(result_landmarks_folder.glob(ext))

            for file in files:
                if file.name.endswith(".npz"):
                    file_name = file.name[:-4]
                elif file.name.endswith(".nii.gz"):
                    file_name = file.name[:-7]
                else:
                    continue
                if case_subset is not None and file_name not in case_subset:
                    continue

                landmarks_pred = process_file(file, original_mask_folder, probabilities_map)
                if len(landmarks_pred.keys()) < 5 and type_set == "six_landmarks":
                    add_info_logging(f"img: {file.name}, not found landmark: {6 - len(landmarks_pred.keys())}",
                                     "result_logger")
                elif len(landmarks_pred.keys()) < 1 and type_set == "gh_landmark":
                    add_info_logging(f"img: {file.name}, not found landmark", "result_logger")

                results = _compute_errors(dict_all_case[file_name], landmarks_pred, results, file_name)

                predictions[file_name] = {
                    point_name[num_key].upper() + "_pred": landmarks_pred[num_key].tolist()
                    for num_key in point_name
                    if num_key != "all" and num_key in landmarks_pred
                }

            if not results:
                return predictions
            # Сохраняем подробный CSV
            results_df = pd.DataFrame(results)
            results_df.to_csv(results_csv_path, index=False)
            with open(predictions_json_path, "w") as f:
                json.dump(predictions, f, indent=2)
            # add_info_logging("finish analysis", "work_logger")

        mean_error = results_df["error"].mean(numeric_only=True)
        std_error = results_df["error"].std(numeric_only=True)
        mean_error_ger_pat = results_df[results_df['group'] == "g"]["error"].mean(numeric_only=True)
        std_error_ger_pat = results_df[results_df['group'] == "g"]["error"].std(numeric_only=True)
        mean_error_slo_pat = results_df[results_df['group'] == "p"]["error"].mean(numeric_only=True)
        std_error_slo_pat = results_df[results_df['group'] == "p"]["error"].std(numeric_only=True)
        mean_error_slo_norm = results_df[results_df['group'] == "n"]["error"].mean(numeric_only=True)
        std_error_slo_norm = results_df[results_df['group'] == "n"]["error"].std(numeric_only=True)

        if type_set == "six_landmarks":
            num_points = sum(1 for k in point_name if k != "all")
            num_img = int(len(results_df["error"]) / num_points)
            not_found = (results_df["error"].isna().sum() / (num_img * num_points)) * 100
            num_img_ger_pat = int(len(results_df[results_df['group'] == "g"]["error"]) / num_points)
            not_found_ger_pat = (results_df[results_df['group'] == "g"]["error"].isna().sum() / (
                    num_img_ger_pat * num_points)) * 100
            num_img_slo_pat = int(len(results_df[results_df['group'] == "p"]["error"]) / num_points)
            not_found_slo_pat = (results_df[results_df['group'] == "p"]["error"].isna().sum() / (
                    num_img_slo_pat * num_points)) * 100
            num_img_slo_norm = int(len(results_df[results_df['group'] == "n"]["error"]) / num_points)
            not_found_slo_norm = (results_df[results_df['group'] == "n"]["error"].isna().sum() / (
                    num_img_slo_norm * num_points)) * 100
        elif type_set == "gh_landmark":
            num_img = int(len(results_df["error"]))
            not_found = (results_df["error"].isna().sum() / num_img) * 100
            num_img_ger_pat = int(len(results_df[results_df['group'] == "g"]["error"]))
            not_found_ger_pat = (results_df[results_df['group'] == "g"]["error"].isna().sum() / num_img_ger_pat) * 100
            num_img_slo_pat = int(len(results_df[results_df['group'] == "p"]["error"]))
            not_found_slo_pat = (results_df[results_df['group'] == "p"]["error"].isna().sum() / num_img_slo_pat) * 100
            num_img_slo_norm = int(len(results_df[results_df['group'] == "n"]["error"]))
            not_found_slo_norm = (results_df[results_df['group'] == "n"]["error"].isna().sum() / num_img_slo_norm) * 100

        data_table = [
            ["All", round(mean_error, 2), round(std_error, 2), round(not_found, 2), num_img],
            ["German\npathology", round(mean_error_ger_pat, 2), round(std_error_ger_pat, 2),
             round(not_found_ger_pat, 2), num_img_ger_pat],
            ["Slovenian\npathology", round(mean_error_slo_pat, 2), round(std_error_slo_pat, 2),
             round(not_found_slo_pat, 2), num_img_slo_pat],
            ["Slovenian\nnormal", round(mean_error_slo_norm, 2), round(std_error_slo_norm, 2),
             round(not_found_slo_norm, 2), num_img_slo_norm],
        ]
        columns = ["Type", "Mean Euclidean\nDistance, mm", "Standard\nDeviation, mm",
                   "Not found, %", "Number of\nimages"]

        results_table_path = result_folder_path / f"landmark_errors_{ds_folder_name}.png"
        plot_table(data_table, columns, results_table_path)

        if type_set == "six_landmarks":
            if active_point_name is not None:
                point_name_dict = {v: v.upper() for k, v in point_name.items() if k != "all"}
                graph_folder = result_folder_path / "exp_landmarks_comparsion"
            else:
                point_name_dict = {"all": "All", "r": "R", "l": "L", "n": "N", "rlc": "RLC", "rnc": "RNC", "lnc": "LNC"}
                graph_folder = result_folder_path / "six_landmarks_comparsion"
                mean_r_error = results_df[results_df['point_id'] == "r"]["error"].mean(numeric_only=True)
                mean_l_error = results_df[results_df['point_id'] == "l"]["error"].mean(numeric_only=True)
                mean_n_error = results_df[results_df['point_id'] == "n"]["error"].mean(numeric_only=True)
                mean_rlc_error = results_df[results_df['point_id'] == "rlc"]["error"].mean(numeric_only=True)
                mean_rnc_error = results_df[results_df['point_id'] == "rnc"]["error"].mean(numeric_only=True)
                mean_lnc_error = results_df[results_df['point_id'] == "lnc"]["error"].mean(numeric_only=True)
            graph_folder.mkdir(parents=True, exist_ok=True)

        elif type_set == "gh_landmark":
            point_name_dict = {"gh": "Geometric Height"}
            graph_folder = result_folder_path / "gh_landmark_comparsion"
            graph_folder.mkdir(parents=True, exist_ok=True)

        for key, type_name in type_label.items():
            err_std = results_df["error"].std(numeric_only=True)
            err_std_groupby = results_df.groupby("group")["error"].std(numeric_only=True)
            if key == "all":
                data_for_plot = results_df[["point_id", "error"]].dropna(how='any')
                plot_group_comparison("point_id", "error", point_name_dict, data_for_plot,
                                      str(graph_folder), type_name)
            else:
                data_for_plot = results_df[results_df['group'] == key][["point_id", "error"]].dropna(how='any')
                plot_group_comparison("point_id","error", point_name_dict, data_for_plot,
                                      str(graph_folder), type_name)

        return predictions

    if find_monte_carlo:
        arr_mean_angles_ger_pat = np.array([]).reshape(0, 3)
        arr_mean_dists_ger_pat = np.array([]).reshape(0, 3)
        arr_mean_angles_slo_pat = np.array([]).reshape(0, 3)
        arr_mean_dists_slo_pat = np.array([]).reshape(0, 3)
        arr_mean_angles_slo_norm = np.array([]).reshape(0, 3)
        arr_mean_dists_slo_norm = np.array([]).reshape(0, 3)
        files = list(result_landmarks_folder.glob("*.npz"))
        for file in files:
            if file.name[0] == "H":
                simulation = LandmarkingMonteCarlo(json_file=str(json_path/file.name[:-4]) + ".json",
                                                   nii_file=str(result_landmarks_folder/file.name[:-4]) + ".nii.gz" ,
                                                   npy_file=str(file))
                cur_angles, cur_dists =  simulation.run_simulation()
                arr_mean_angles_ger_pat = np.vstack([arr_mean_angles_ger_pat, cur_angles])
                arr_mean_dists_ger_pat = np.vstack([arr_mean_dists_ger_pat, cur_dists])
            if file.name[0] == "p":
                simulation = LandmarkingMonteCarlo(json_file=str(json_path / file.name[:-4]) + ".json",
                                                   nii_file=str(result_landmarks_folder / file.name[:-4]) + ".nii.gz",
                                                   npy_file=str(file))
                cur_angles, cur_dists = simulation.run_simulation()
                arr_mean_angles_slo_pat = np.vstack([arr_mean_angles_slo_pat, cur_angles])
                arr_mean_dists_slo_pat = np.vstack([arr_mean_dists_slo_pat, cur_dists])
            if file.name[0] == "n":
                simulation = LandmarkingMonteCarlo(json_file=str(json_path / file.name[:-4]) + ".json",
                                                   nii_file=str(result_landmarks_folder / file.name[:-4]) + ".nii.gz",
                                                   npy_file=str(file))
                cur_angles, cur_dists = simulation.run_simulation()
                arr_mean_angles_slo_norm = np.vstack([arr_mean_angles_slo_norm, cur_angles])
                arr_mean_dists_slo_norm = np.vstack([arr_mean_dists_slo_norm, cur_dists])
        add_info_logging("German pathology")
        add_info_logging(f"mean angles = '{np.mean(arr_mean_angles_ger_pat, axis=0)}'")
        add_info_logging(f"mean distances = '{np.mean(arr_mean_dists_ger_pat, axis=0)}'")
        add_info_logging("Slovenian pathology")
        add_info_logging(f"mean angles = '{np.mean(arr_mean_angles_slo_pat, axis=0)}'")
        add_info_logging(f"mean distances = '{np.mean(arr_mean_dists_slo_pat, axis=0)}'")
        add_info_logging("Slovenian normal")
        add_info_logging(f"mean angles = '{np.mean(arr_mean_angles_slo_norm, axis=0)}'")
        add_info_logging(f"mean distances = '{np.mean(arr_mean_dists_slo_norm, axis=0)}'")
        add_info_logging("Sum")
        arr_mean_angles = np.vstack([arr_mean_angles_ger_pat, arr_mean_angles_slo_pat, arr_mean_angles_slo_norm])
        arr_mean_dists = np.vstack([arr_mean_dists_ger_pat, arr_mean_dists_slo_pat, arr_mean_dists_slo_norm])
        add_info_logging(f"mean angles = '{np.mean(arr_mean_angles, axis=0)}'")
        add_info_logging(f"mean distances = '{np.mean(arr_mean_dists, axis=0)}'")


def find_morphometric_parameters(data_path, ds_folder_name):
    data_path = Path(data_path)
    result_landmarks_folder = data_path / "nnUNet_folder" / "nnUNet_test" / ds_folder_name
    original_mask_folder = data_path / "nnUNet_folder" / "original_mask" / ds_folder_name
    result_folder_path = data_path / "result"
    results_csv_path = result_folder_path / f"morphometric_parameters_{ds_folder_name}.csv"
    files = list(result_landmarks_folder.glob("*.npz"))
    summary_result = []
    for file in files:
        lcc = LandmarkCentersCalculator()
        file_name = file.name[:-4] + ".nii.gz"
        landmarks_centers_of_peak = lcc.extract_landmarks_peak_npz(original_mask_folder / file_name, file)
        landmarks_centers_of_mass = lcc.extract_landmarks_com_npz(original_mask_folder / file_name, file)
        landmarks_centers_of_peaks_topk = lcc.extract_landmarks_topk_peaks_npz(original_mask_folder / file_name,
                                                                               file, top_k=10)
        for metric, sets_landmarks in metric_to_landmarks.items():
            result_cop = controller_metrics(metric, sets_landmarks, landmarks_centers_of_peak)
            result_com = controller_metrics(metric, sets_landmarks, landmarks_centers_of_mass)
            result_mc = controller_metrics(metric, sets_landmarks, landmarks_centers_of_peaks_topk, mc_option=True)
            summary_result.append({
                "file": file_name,
                "metric": metric,
                "peak": result_cop,
                "center_of_mass": result_com,
                "topk_peaks_avg": result_mc
            })

    df = pd.DataFrame.from_records(summary_result)
    result_folder_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_csv_path, index=False)

