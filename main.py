import os
import logging
import shutil
from pathlib import Path
import platform
import json

import numpy as np
import pandas as pd
from datetime import datetime

from configurator.equipment_analysis import get_free_cpus
from data_preprocessing.dcm_nii_converter import convert_dcm_to_nii, check_dcm_info
from data_preprocessing.stl_nii_converter import convert_and_cut_stl
from data_preprocessing.check_structure import create_directory_structure
from data_preprocessing.text_worker import (json_reader, yaml_reader, yaml_save, json_save,
                                            add_info_logging, create_new_json, parse_txt_file)
from data_preprocessing.csv_worker import write_csv, read_csv
from data_preprocessing.crop_nii import cropped_image, find_global_size
# from data_postprocessing.evaluation_analysis import landmarking_testing
from data_postprocessing.controller_analysis import (landmarks_analysis, experiment_analysis, mask_analysis,
                                                     find_morphometric_parameters, LandmarkCentersCalculator,
                                                     curve_lines_analysis, basal_ring_analysis)
from data_postprocessing.coherent_point_drift import create_new_gh_json, find_new_curv
from data_visualization.markers import (slices_with_markers, process_markers, find_mean_gh_landmark,
                                        process_mask_curve_lines)
from models.controller_nnUnet import process_nnunet
from experiments.nnUnet_experiments import experiment_training
from models.controller_GNN import process_gnn, process_gnn_interobserver
from data_preprocessing.oblique_slice_extractor import find_global_2d_size, extract_2d_slice_pair, compute_br_plane_offset

# from optimization.parallelization import division_processes


def _cleanup_empty_logs(log_folder):
    log_dir = Path(log_folder)
    if not log_dir.exists():
        return
    for log_file in log_dir.iterdir():
        if log_file.is_file() and log_file.stat().st_size == 0:
            log_file.unlink()


def controller(data_path, cpus):
    def _find_series_folders(root_folder, types_file, parent=False):
        if isinstance(types_file, str):
            types_file = [types_file]
        series_folders = set()
        for ext in types_file:
            for file_path in Path(root_folder).rglob(f"*.{ext}"):
                if parent:
                    series_folders.add(file_path.parent)
                else:
                    series_folders.add(file_path)
        return list(series_folders)

    def clear_folder(folder_path):
        """Очищает папку, удаляя все файлы и подпапки"""
        folder = Path(folder_path)
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            add_info_logging(f"The folder '{folder_path}' did not exist, so it was created.",
                             "work_logger")
            return

        for item in folder.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()  # Удаляем файл или символическую ссылку
            elif item.is_dir():
                shutil.rmtree(item)  # Удаляем папку рекурсивно

    temp_path = os.path.join(data_path, "temp")

    result_folder = os.path.join(data_path, "result")
    dicom_folder = os.path.join(data_path, "dicom")
    image_folder = os.path.join(data_path, "image_nii")
    image_crop_folder = os.path.join(data_path, "image_nii_crop")
    txt_points_folder = os.path.join(data_path, "txt_points")
    stl_aorta_segment_folder = os.path.join(data_path, "stl_aorta_segment")
    nnUNet_folder = os.path.join(data_path, "nnUNet_folder")
    gnn_folder = os.path.join(data_path, "gnn_folder")
    gnn_interobserver_folder = os.path.join(data_path, "gnn_interobserver_study")
    interobserver_txt_points_folder = os.path.join(data_path, "interobserver_txt_points")
    mask_aorta_segment_folder = os.path.join(data_path, "mask_aorta_segment")
    mask_aorta_segment_crop_folder = os.path.join(data_path, "mask_aorta_segment_crop")
    image_br_2d_folder = os.path.join(data_path, "image_basal_ring_2d")
    mask_br_2d_folder = os.path.join(data_path, "mask_basal_ring_2d")
    mask_6_landmarks_folder = os.path.join(data_path, "mask_6_landmarks")
    mask_gh_landmark_folder = os.path.join(data_path, "mask_gh_landmark")
    mask_gh_lines_folder = os.path.join(data_path, "mask_gh_lines")
    mask_ci_lines_folder = os.path.join(data_path, "mask_ci_lines")

    json_marker_folder = os.path.join(data_path, "json_markers_info")
    json_info_folder = os.path.join(data_path, "json_info")
    json_duplication_g_h_path = os.path.join(data_path, "json_duplication_geometric_heights")
    json_land_mask_coord_path = os.path.join(data_path, "json_landmarks_mask_coord")

    controller_path = os.path.join(script_dir, "controller.yaml")
    dict_all_case_path = os.path.join(data_path, "dict_all_case.json")

    dict_all_case = {}
    if os.path.isfile(controller_path):
        controller_dump = yaml_reader(controller_path)
    else:
        controller_dump = {}

    if not controller_dump.get("check_metadata_dicom"):
        summary_info = []
        dicom_folders = _find_series_folders(dicom_folder, "dcm", parent=True)

        for folder_path in dicom_folders:
            case_name = folder_path.name
            info = check_dcm_info(folder_path)
            info["case_name"] = case_name
            summary_info.append(info)
        df = pd.DataFrame(summary_info)
        write_csv(df, result_folder, "dicom_metadata_info.csv")

        # 📄 Сохраняем JSON с уникальными значениями по каждому столбцу (кроме case_name)
        unique_info = {
            col: sorted([str(x) for x in df[col].dropna().unique().tolist()])
            for col in df.columns if col != "case_name"
        }

        # Преобразуем к сериализуемым типам
        def convert(o):
            if isinstance(o, (tuple, list)):
                return list(o)
            return str(o)

        json_path = os.path.join(result_folder, "dicom_metadata_info_summary.json")
        with open(json_path, "w") as f:
            json.dump(unique_info, f, indent=4, default=convert)

        controller_dump["check_metadata_dicom"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("create_train_test_lists"):
        df = read_csv(result_folder, "dicom_metadata_info.csv")

        result_rows = []

        h_cases = df[df['case_name'].str.startswith('H')].copy()
        np_cases = df[df['case_name'].str.startswith(('n', 'p'))].copy()

        for idx, row in enumerate(h_cases.itertuples(index=False), start=1):
            result_rows.append({
                'case_name': row.case_name,
                'used_case_name': f'g{idx}',
                'type_series': 'test'
            })

        for letter in ['n', 'p']:
            group = np_cases[np_cases['case_name'].str.startswith(letter)].copy()
            n_total = len(group)
            n_test = int(round(n_total * 0.1))  # 10%

            shuffled = group.sample(frac=1, random_state=42).reset_index(drop=True)
            shuffled['type_series'] = ['test'] * n_test + ['train'] * (n_total - n_test)
            shuffled['used_case_name'] = shuffled['case_name']

            result_rows.extend(shuffled[['case_name', 'used_case_name', 'type_series']].to_dict(orient='records'))

        result_df = pd.DataFrame(result_rows)
        write_csv(result_df, result_folder, "train_test_lists.csv")

        controller_dump["create_train_test_lists"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("create_new_train_test_lists"):
        df = pd.read_csv(f"{result_folder}/train_test_lists.csv")

        # 1. для случайных 80% кейсов с used_case_name начинающимся на 'g' → train
        g_mask = df['used_case_name'].str.startswith('g')
        g_cases = df[g_mask].sample(frac=0.8, random_state=42)  # 80%
        df.loc[g_cases.index, 'type_series'] = 'train'

        # 2. для случайных 15 кейсов с used_case_name начинающимся на 'n' и имеющих type_series == 'train' → test
        n_mask = (df['used_case_name'].str.startswith('n')) & (df['type_series'] == 'train')
        n_cases = df[n_mask].sample(n=min(15, n_mask.sum()), random_state=42)
        df.loc[n_cases.index, 'type_series'] = 'test'

        # 3. для случайных 6 кейсов с used_case_name начинающимся на 'p' и имеющих type_series == 'train' → test
        p_mask = (df['used_case_name'].str.startswith('p')) & (df['type_series'] == 'train')
        p_cases = df[p_mask].sample(n=min(6, p_mask.sum()), random_state=42)
        df.loc[p_cases.index, 'type_series'] = 'test'

        # сохраняем в новый файл
        df.to_csv(f"{result_folder}/train_test_lists_2.csv", index=False)

        controller_dump["create_new_train_test_lists"] = True
        yaml_save(controller_dump, controller_path)

    train_test_lists = read_csv(result_folder, "train_test_lists.csv")
    test_cases = train_test_lists[train_test_lists['type_series'] == 'test']['used_case_name'].tolist()
    train_cases = train_test_lists[train_test_lists['type_series'] == 'train']['used_case_name'].tolist()

    if not controller_dump.get("convert_resample_dicom"):
        clear_folder(image_folder)
        dicom_folders = _find_series_folders(dicom_folder, "dcm", parent=True)

        for folder_path in dicom_folders:
            org_folder_name = folder_path.name
            try:
                new_file_name = train_test_lists.loc[
                    train_test_lists['case_name'] == org_folder_name, 'used_case_name'
                ].values[0]
            except IndexError:
                add_info_logging(f"'{org_folder_name}' not found in CSV.")
                continue

            img_convert_path = os.path.join(image_folder, new_file_name)
            convert_dcm_to_nii(str(folder_path), img_convert_path, zip=True)

        controller_dump["convert_resample_dicom"] = True
        yaml_save(controller_dump, controller_path)

    if not dict_all_case:
        if os.path.isfile(dict_all_case_path):
            dict_all_case = json_reader(dict_all_case_path)
        else:
            controller_dump["create_dict_all_case"] = False
            yaml_save(controller_dump, controller_path)
            add_info_logging("There is no data dictionary for all cases. The dictionary will be regenerated.")

    if not controller_dump.get("create_dict_all_case"):
        txt_files = _find_series_folders(txt_points_folder, "txt")
        dict_all_case = {}

        for txt_path in txt_files:
            case_base_name = txt_path.stem[2:]  # Удаляем первые 2 символа

            if case_base_name[0] == "H":
                case_base_name = case_base_name + "_MJ"
            # Найти в датафрейме
            match = train_test_lists[train_test_lists["case_name"] == case_base_name]
            if match.empty:
                add_info_logging(f" Not found: {case_base_name} in dict cases")
                continue

            used_case_name = match.iloc[0]["used_case_name"]
            type_series = match.iloc[0]["type_series"]

            parsed_data = parse_txt_file(txt_path)
            parsed_data["type_series"] = type_series

            dict_all_case[used_case_name] = parsed_data

        json_save(dict_all_case, dict_all_case_path)
        controller_dump["create_dict_all_case"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("mask_aorta_segment"):
        stl_files = _find_series_folders(stl_aorta_segment_folder, "stl")
        clear_folder(mask_aorta_segment_folder)
        for stl_path in stl_files:
            case_base_name = stl_path.stem

            if case_base_name[0] == "H":
                case_base_name = case_base_name + "_MJ"
            match = train_test_lists[train_test_lists["case_name"] == case_base_name]
            if match.empty:
                add_info_logging(f" Not found: {case_base_name} in dict cases")
                continue

            used_case_name = match.iloc[0]["used_case_name"]

            mask_aorta_segment_path = os.path.join(mask_aorta_segment_folder, f"{used_case_name}.nii.gz")
            image_path = os.path.join(image_folder, f"{used_case_name}.nii.gz")
            nadir_points = [dict_all_case[used_case_name]['R'][0],
                            dict_all_case[used_case_name]['L'][0],
                            dict_all_case[used_case_name]['N'][0]]
            commissural_points = [dict_all_case[used_case_name]['RLC'][0],
                                  dict_all_case[used_case_name]['RNC'][0],
                                  dict_all_case[used_case_name]['LNC'][0]]
            convert_and_cut_stl(stl_path, image_path, mask_aorta_segment_path,
                                nadir_points, commissural_points, margin=2)
            controller_dump["mask_aorta_segment"] = True
            yaml_save(controller_dump, controller_path)

    if not controller_dump.get("nnUNet_aorta_segment_train"):
        dict_dataset = {
            "channel_names": {0: "CT"},
            "labels": {'background': 0, 'aortic_valve': 1},
            "file_ending": ".nii.gz"
        }
        num = controller_dump["number_aorta_segment"]
        name = controller_dump["name_aorta_segment"]
        process_nnunet(folder=nnUNet_folder, ds_folder_name=f"Dataset{num}_{name}", id_case=num,
                       folder_image_path=image_folder, folder_mask_path=mask_aorta_segment_folder,
                       dict_dataset=dict_dataset, train_test_lists=train_test_lists,
                       create_ds=True, training_mod=True, predicting_mod=True)
        controller_dump["nnUNet_aorta_segment_train"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("crop_img_size"):
        all_image_paths = [
            os.path.join(mask_aorta_segment_folder, f)
            for f in os.listdir(mask_aorta_segment_folder)
            if os.path.isfile(os.path.join(mask_aorta_segment_folder, f))
        ]

        padding = 10
        # Найти общий bounding box для всех изображений
        global_size = find_global_size(all_image_paths, padding)
        controller_dump["crop_img_size"] = [round(x) for x in global_size]
        yaml_save(controller_dump, controller_path)
    else:
        global_size = controller_dump["crop_img_size"]

    if not controller_dump.get("crop_images"):
        clear_folder(os.path.join(image_crop_folder))

        for case in os.listdir(os.path.join(image_folder)):
            cropped_image(mask_image_path=str(os.path.join(mask_aorta_segment_folder, case)),
                          input_image_path=str(os.path.join(image_folder, case)),
                          output_image_path=str(os.path.join(image_crop_folder, case)),
                          size=global_size)
        controller_dump["crop_images"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("crop_mask_aorta_segment"):
        clear_folder(mask_aorta_segment_crop_folder)

        for case in os.listdir(mask_aorta_segment_folder):
            mask_path = str(os.path.join(mask_aorta_segment_folder, case))
            cropped_image(mask_image_path=mask_path,
                          input_image_path=mask_path,
                          output_image_path=str(os.path.join(mask_aorta_segment_crop_folder, case)),
                          size=global_size)
        controller_dump["crop_mask_aorta_segment"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("extract_2d_slices"):

        if not controller_dump.get("crop_br_2d_size"):
            global_2d_size = find_global_2d_size(
                mask_crop_folder=mask_aorta_segment_crop_folder,
                dict_all_case=dict_all_case,
                initial_size=512,
                padding=10
            )
            controller_dump["crop_br_2d_size"] = global_2d_size
            yaml_save(controller_dump, controller_path)
        else:
            global_2d_size = controller_dump["crop_br_2d_size"]

        clear_folder(image_br_2d_folder)
        clear_folder(mask_br_2d_folder)

        for case_name, points_dict in dict_all_case.items():
            image_3d_path = os.path.join(image_crop_folder, f"{case_name}.nii.gz")
            mask_3d_path = os.path.join(mask_aorta_segment_crop_folder, f"{case_name}.nii.gz")

            if not os.path.isfile(image_3d_path) or not os.path.isfile(mask_3d_path):
                add_info_logging(f"Skipping {case_name}: 3D file not found.", "work_logger")
                continue

            r, l, n = points_dict['R'][0], points_dict['L'][0], points_dict['N'][0]

            extract_2d_slice_pair(
                image_3d_path=image_3d_path,
                mask_3d_path=mask_3d_path,
                image_2d_path=os.path.join(image_br_2d_folder, f"{case_name}.nii.gz"),
                mask_2d_path=os.path.join(mask_br_2d_folder, f"{case_name}.nii.gz"),
                r=r, l=l, n=n,
                size_hw=global_2d_size
            )

        json_save(dict_all_case, dict_all_case_path)
        controller_dump["extract_2d_slices"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("br_plane_offset"):
        offset_rows = []
        for case_name, points_dict in dict_all_case.items():
            if "BR - closed" not in points_dict:
                continue
            r, l, n = points_dict['R'][0], points_dict['L'][0], points_dict['N'][0]
            offset = compute_br_plane_offset(points_dict["BR - closed"], r=r, l=l, n=n)
            dict_all_case[case_name]["BR - plane offset"] = offset
            offset_rows.append({"case": case_name, "BR - plane offset": offset})

        csv_path = os.path.join(result_folder, "br_plane_offset.csv")
        if os.path.exists(csv_path):
            os.remove(csv_path)
        pd.DataFrame(offset_rows).to_csv(csv_path, index=False)

        total = len(offset_rows)
        shifted = sum(1 for row in offset_rows if row["BR - plane offset"] != 0.0)
        if total > 0:
            add_info_logging(
                f"BR plane offset: {shifted}/{total} cases with non-zero offset "
                f"({round(100 * shifted / total, 1)}%)",
                "result_logger"
            )

        json_save(dict_all_case, dict_all_case_path)
        controller_dump["br_plane_offset"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("nnUNet_br_2d_train"):
        dict_dataset = {
            "channel_names": {0: "CT"},
            "labels": {'background': 0, 'aortic_valve': 1},
            "file_ending": ".nii.gz"
        }
        num = controller_dump["number_br_2d"]
        name = controller_dump["name_br_2d"]
        process_nnunet(folder=nnUNet_folder, ds_folder_name=f"Dataset{num}_{name}", id_case=num,
                       folder_image_path=image_br_2d_folder, folder_mask_path=mask_br_2d_folder,
                       dict_dataset=dict_dataset, train_test_lists=train_test_lists,
                       create_ds=True, training_mod=True, predicting_mod=True, network="2d")
        controller_dump["nnUNet_br_2d_train"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("analys_result_br_2d"):
        num = controller_dump["number_br_2d"]
        name = controller_dump["name_br_2d"]
        predictions = basal_ring_analysis(data_path, result_folder, folder_name=f"Dataset{num}_{name}",
                                          dict_cases=dict_all_case, plane_offset_mode="key",
                                          n_curve_points=20)
        if predictions:
            for case_name, pred_coords in predictions.items():
                if case_name in dict_all_case:
                    dict_all_case[case_name].update(pred_coords)
            json_save(dict_all_case, dict_all_case_path)
        controller_dump["analys_result_br_2d"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("nnUNet_br_2d_pred_plane"):
        num = controller_dump["number_br_2d"]
        name = controller_dump["name_br_2d"]
        ds_folder_name = f"Dataset{num}_{name}"

        images_ts_path = Path(nnUNet_folder) / "nnUNet_raw" / ds_folder_name / "imagesTs"
        images_ts_true_path = Path(nnUNet_folder) / "nnUNet_raw" / ds_folder_name / "imagesTs_true_plane"

        if images_ts_path.exists() and not images_ts_true_path.exists():
            images_ts_path.rename(images_ts_true_path)
        images_ts_path.mkdir(exist_ok=True)

        global_2d_size = controller_dump["crop_br_2d_size"]
        test_cases = train_test_lists[train_test_lists['type_series'] == 'test']['used_case_name'].tolist()

        for case_name in test_cases:
            points_dict = dict_all_case.get(case_name, {})
            if not all(k in points_dict for k in ('R_pred', 'L_pred', 'N_pred')):
                add_info_logging(f"Skipping {case_name}: predicted landmarks not found.", "work_logger")
                continue
            image_3d_path = os.path.join(image_crop_folder, f"{case_name}.nii.gz")
            if not os.path.isfile(image_3d_path):
                add_info_logging(f"Skipping {case_name}: 3D image not found.", "work_logger")
                continue
            extract_2d_slice_pair(
                image_3d_path=image_3d_path,
                image_2d_path=str(images_ts_path / f"{case_name}_0000.nii.gz"),
                r=points_dict['R_pred'],
                l=points_dict['L_pred'],
                n=points_dict['N_pred'],
                size_hw=global_2d_size
            )

        process_nnunet(folder=nnUNet_folder, ds_folder_name=ds_folder_name, id_case=num,
                       folder_image_path=image_br_2d_folder, folder_mask_path=mask_br_2d_folder,
                       dict_dataset={"channel_names": {0: "CT"},
                                     "labels": {'background': 0, 'aortic_valve': 1},
                                     "file_ending": ".nii.gz"},
                       train_test_lists=train_test_lists,
                       create_ds=False, training_mod=False, predicting_mod=True, network="2d")

        controller_dump["nnUNet_br_2d_pred_plane"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("mask_6_landmarks"):
        clear_folder(os.path.join(mask_6_landmarks_folder))
        for case_name, points_dict in dict_all_case.items():
            process_markers(image_path=os.path.join(image_crop_folder, f"{case_name}.nii.gz"),
                            dict_case=points_dict,
                            output_path=os.path.join(mask_6_landmarks_folder, f"{case_name}.nii.gz"),
                            radius=9, keys_to_need={'R': 1, 'L': 2, 'N': 3, 'RLC': 4, 'RNC': 5, 'LNC': 6})
        controller_dump["mask_6_landmarks"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("nnUNet_6_landmarks_train"):
        dict_dataset = {
            "channel_names": {0: "CT"},
            "labels": {"background": 0, "R": 1, "L": 2, "N": 3, "RLC": 4, "RNC": 5, "LNC": 6},
            "file_ending": ".nii.gz"
        }
        num = controller_dump["number_6_landmarks"]
        name = controller_dump["name_6_landmarks"]
        process_nnunet(folder=nnUNet_folder, ds_folder_name=f"Dataset{num}_{name}", id_case=num,
                       folder_image_path=image_crop_folder, folder_mask_path=mask_6_landmarks_folder,
                       dict_dataset=dict_dataset, train_test_lists=train_test_lists,
                       create_ds=True, training_mod=True, predicting_mod=True)
        controller_dump["nnUNet_6_landmarks_train"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("find_and_create_mask_gh_landmark"):
        clear_folder(mask_gh_landmark_folder)
        for case_name, points_dict in dict_all_case.items():
            dict_all_case[case_name]["GH"] = [find_mean_gh_landmark(points_dict)]
            process_markers(image_path=os.path.join(image_crop_folder, f"{case_name}.nii.gz"),
                            dict_case=points_dict,
                            output_path=os.path.join(mask_gh_landmark_folder, f"{case_name}.nii.gz"),
                            radius=9, keys_to_need={'GH': 1})
        json_save(dict_all_case, dict_all_case_path)
        controller_dump["find_and_create_mask_gh_landmark"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("nnUNet_gh_landmark_train"):
        dict_dataset = {
            "channel_names": {0: "CT"},
            "labels": {'background': 0, 'gh': 1},
            "file_ending": ".nii.gz"
        }
        num = controller_dump["number_gh_landmark"]
        name = controller_dump["name_gh_landmark"]
        process_nnunet(folder=nnUNet_folder, ds_folder_name=f"Dataset{num}_{name}", id_case=num,
                       folder_image_path=image_crop_folder, folder_mask_path=mask_gh_landmark_folder,
                       dict_dataset=dict_dataset, train_test_lists=train_test_lists,
                       create_ds=True, training_mod=True, predicting_mod=True)
        controller_dump["nnUNet_gh_landmark_train"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump["mask_gh_lines"]:
        clear_folder(os.path.join(mask_gh_lines_folder))
        for case_name, points_dict in dict_all_case.items():
            process_mask_curve_lines(image_path=os.path.join(image_crop_folder, f"{case_name}.nii.gz"),
                                  dict_case=points_dict,
                                  output_path=os.path.join(mask_gh_lines_folder, f"{case_name}.nii.gz"),
                                  radius=2, keys_to_need={'RGH': 1, 'LGH': 2, 'NGH': 3})
        controller_dump["mask_gh_lines"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("nnUNet_gh_lines_train"):
        dict_dataset = {
            "channel_names": {0: "CT"},
            "labels": {'background': 0, 'RGH': 1, 'LGH': 2, 'NGH': 3},
            "file_ending": ".nii.gz"
        }
        num = controller_dump["number_gh_lines"]
        name = controller_dump["name_gh_lines"]
        process_nnunet(folder=nnUNet_folder, ds_folder_name=f"Dataset{num}_{name}", id_case=num,
                       folder_image_path=image_crop_folder, folder_mask_path=mask_gh_lines_folder,
                       dict_dataset=dict_dataset, train_test_lists=train_test_lists,
                       create_ds=True, training_mod=True, predicting_mod=True)
        controller_dump["nnUNet_gh_lines_train"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("mask_ci_lines"):
        clear_folder(os.path.join(mask_ci_lines_folder))
        for case_name, points_dict in dict_all_case.items():
            process_mask_curve_lines(image_path=os.path.join(image_crop_folder, f"{case_name}.nii.gz"),
                                  dict_case=points_dict,
                                  output_path=os.path.join(mask_ci_lines_folder, f"{case_name}.nii.gz"),
                                  radius=2, keys_to_need={'RCI': 1, 'LCI': 2, 'NCI': 3})
        controller_dump["mask_ci_lines"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("nnUNet_ci_lines_train"):
        dict_dataset = {
            "channel_names": {0: "CT"},
            "labels": {'background': 0, 'RCI': 1, 'LCI': 2, 'NCI': 3},
            "file_ending": ".nii.gz"
        }
        num = controller_dump["number_ci_lines"]
        name = controller_dump["name_ci_lines"]
        process_nnunet(folder=nnUNet_folder, ds_folder_name=f"Dataset{num}_{name}", id_case=num,
                       folder_image_path=image_crop_folder, folder_mask_path=mask_ci_lines_folder,
                       dict_dataset=dict_dataset, train_test_lists=train_test_lists,
                       create_ds=True, training_mod=True, predicting_mod=True)
        controller_dump["nnUNet_ci_lines_train"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("gnn_metrics_train_test"):
        num = controller_dump["number_6_landmarks"]
        name = controller_dump["name_6_landmarks"]
        result_nnunet_landmarks_folder = os.path.join(nnUNet_folder, "nnUNet_test", f"Dataset{num}_{name}")
        process_gnn(result_6_nnunet_folder=result_nnunet_landmarks_folder, gnn_folder=gnn_folder,
                    train_test_lists=train_test_lists, json_info_folder=json_info_folder,
                    create_ds=False, training_mod=False, testing_mod=True)
        controller_dump["gnn_metrics_train_test"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("gnn_interobserver_study"):
        process_gnn_interobserver(gnn_folder=gnn_interobserver_folder,
                                   json_info_folder=json_info_folder,
                                   interobserver_txt_folder=interobserver_txt_points_folder,
                                   create_ds=True, testing_mod=True)
        controller_dump["gnn_interobserver_study"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump["duplication_geometric_heights"]:
        ds_folder_name = "Dataset489_AortaLandmarks"
        train_nnUNet_DS_landmarks_path = os.path.join(nnUNet_folder, "nnUNet_raw", ds_folder_name, "imagesTr")
        test_nnUNet_DS_landmarks_path = os.path.join(nnUNet_folder, "nnUNet_raw", ds_folder_name, "imagesTs")
        cases_folders_list = [train_nnUNet_DS_landmarks_path, test_nnUNet_DS_landmarks_path]
        json_dupl_folder = os.path.join(json_duplication_g_h_path, "train")
        for cur_folder in cases_folders_list:
            clear_folder(json_dupl_folder)
            for file in os.listdir(cur_folder):
                file_name = file[:-12]
                first_letter = file[:1]
                if first_letter == "H":
                    json_org_file = os.path.join(json_marker_folder, "Homburg pathology", f"{file_name}.json")
                elif first_letter == "n":
                    json_org_file = os.path.join(json_marker_folder, "Normal", f"{file_name}.json")
                else:
                    json_org_file = os.path.join(json_marker_folder, "Pathology", f"{file_name}.json")
                json_dupl_file = os.path.join(json_dupl_folder, f"{file_name}.json")

                create_new_gh_json(json_org_file, json_dupl_file, n_points=10)
            json_dupl_folder = os.path.join(json_duplication_g_h_path, "test")

        controller_dump["duplication_geometric_heights"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump["json_landmarks_mask_coord"]:
        ds_folder_name_ldms = "Dataset489_AortaLandmarks"
        ds_folder_name_gh = "Dataset479_GeometricHeight"
        cur_nnUNet_folder_path = Path(nnUNet_folder)
        train_mask_ldms_folder_path = cur_nnUNet_folder_path / "nnUNet_raw" / ds_folder_name_ldms / "labelsTr"
        test_mask_ldms_folder_path = cur_nnUNet_folder_path / "nnUNet_test" / ds_folder_name_ldms
        original_mask_ldms_folder_path = cur_nnUNet_folder_path / "original_mask" / ds_folder_name_ldms
        mask_gh_train_folder = cur_nnUNet_folder_path / "nnUNet_raw" / ds_folder_name_gh / "labelsTr"
        mask_gh_org_folder = cur_nnUNet_folder_path / "original_mask" / ds_folder_name_gh
        mask_gh_test_folder = cur_nnUNet_folder_path / "nnUNet_test" / ds_folder_name_gh
        cases_folders_list = [train_mask_ldms_folder_path, test_mask_ldms_folder_path]

        type_set = "train"
        res_test = LandmarkCentersCalculator()
        json_land_mask_org_coord_folder = os.path.join(json_land_mask_coord_path, "original")
        for cur_folder in cases_folders_list:
            if type_set == "train":
                files = list(Path(cur_folder).glob("*.nii.gz"))
                json_land_mask_coord_folder = os.path.join(json_land_mask_coord_path, "train")
            else:
                files = list(Path(cur_folder).glob("*.npz"))
                json_land_mask_coord_folder = os.path.join(json_land_mask_coord_path, "test")
            for file in files:
                json_dupl_file = os.path.join(json_land_mask_coord_folder, f"{file.name.split('.')[0]}.json")
                pred_org = {}
                file_name = file.name.split('.')[0] + ".nii.gz"
                if type_set == "train":
                    mask_gh_org_file = mask_gh_train_folder / file_name
                    labels = {1: "R", 2: "L", 3: "N", 4: "RLC", 5: "RNC", 6: "LNC"}
                    pred = {}
                    for key, name in labels.items():
                        pred[key] = dict_all_case[file.name.split('.')[0]][name]
                    pred_gh_train = res_test.extract_landmarks_com_nii(mask_gh_org_file)
                    pred[7] = pred_gh_train[1]
                else:
                    mask_gh_org_file = mask_gh_org_folder / file_name
                    mask_gh_test_file = mask_gh_test_folder / str(file.name.split('.')[0] + ".npz")
                    pred = res_test.extract_landmarks_com_npz(original_mask_ldms_folder_path / file_name, file)
                    labels = {1: "R", 2: "L", 3: "N", 4: "RLC", 5: "RNC", 6: "LNC"}
                    for key, name in labels.items():
                        pred_org[key] = dict_all_case[file.name.split('.')[0]][name]
                    pred_gh_test = res_test.extract_landmarks_com_npz(original_mask_ldms_folder_path / file_name,
                                                                      mask_gh_test_file)
                    pred[7] = pred_gh_test[1]
                    pred_gh_org = res_test.extract_landmarks_com_nii(mask_gh_org_file)
                    pred_org[7] = pred_gh_org[1]
                create_new_json(json_dupl_file, pred)
                if pred_org:
                    json_dupl_org_file = os.path.join(json_land_mask_org_coord_folder, f"{file.name.split('.')[0]}.json")
                    create_new_json(json_dupl_org_file, pred_org)
            type_set = "test"

        controller_dump["json_landmarks_mask_coord"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump["find_geometric_heights"]:
        test_cases_path = os.path.join(json_land_mask_coord_path, "test")
        train_cases_path = os.path.join(json_land_mask_coord_path, "train")
        org_cases_path = os.path.join(json_land_mask_coord_path, "original")
        list_test_cases = list(Path(test_cases_path).glob("*.json"))
        list_train_cases = list(Path(train_cases_path).glob("*.json"))
        test_curv_cases_path = os.path.join(json_duplication_g_h_path, "test")
        train_curv_cases_path = os.path.join(json_duplication_g_h_path, "train")
        result_curv_cases_path = os.path.join(json_duplication_g_h_path, "result")
        clear_folder(result_curv_cases_path)

        errors = []  # сюда будем собирать результаты по кейсам

        for cur_test_case in list_test_cases:
            find_new_curv(cur_test_case, list_train_cases, train_curv_cases_path, result_curv_cases_path, org_cases_path)

            # Сравнение предсказанных кривых с истинными
            pred_json_path = Path(result_curv_cases_path) / cur_test_case.name
            true_json_path = Path(test_curv_cases_path) / cur_test_case.name

            def _load_curves(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                return {k: np.array(v) for k, v in data.items()}

            pred_curves = _load_curves(pred_json_path)
            true_curves = _load_curves(true_json_path)

            combined_pred = []
            combined_true = []

            for name in ["RGH", "NGH", "LGH"]:

                pred = pred_curves[name]
                true = true_curves[name]

                combined_pred.append(pred)
                combined_true.append(true)

            if combined_pred and combined_true:
                all_pred = np.vstack(combined_pred)
                all_gt = np.vstack(combined_true)

                diff = all_pred - all_gt
                mse = np.mean(np.square(diff))
                rmse = np.sqrt(mse)

                # Сохраняем ошибку в список
                errors.append({"case": cur_test_case.name.split('.')[0], "rmse": rmse})

        # После цикла создаем DataFrame
        df_errors = pd.DataFrame(errors)

        # Вычисляем среднее и std
        mean_rmse = df_errors["rmse"].mean()
        std_rmse = df_errors["rmse"].std()

        print(f"Average RMSE across all cases: {mean_rmse:.3f}")
        print(f"Standard Deviation RMSE: {std_rmse:.3f}")

        # Сохраняем в CSV
        csv_path = Path(result_folder) / "rmse_errors_per_case.csv"
        df_errors.to_csv(csv_path, index=False)

    if not controller_dump.get("analys_result_aorta_segment"):
        num = controller_dump["number_aorta_segment"]
        name = controller_dump["name_aorta_segment"]
        mask_analysis(data_path, result_folder, type_mask="aortic_valve", folder_name=f"Dataset{num}_{name}")
        controller_dump["analys_result_aorta_segment"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("analys_result_6_landmarks"):
        num = controller_dump["number_6_landmarks"]
        name = controller_dump["name_6_landmarks"]
        predictions = landmarks_analysis(Path(data_path), dict_all_case, ds_folder_name=f"Dataset{num}_{name}",
                                         find_center_mass=True, probabilities_map=True)
        if predictions:
            for case_name, pred_coords in predictions.items():
                if case_name in dict_all_case:
                    dict_all_case[case_name].update(pred_coords)
            json_save(dict_all_case, dict_all_case_path)
        controller_dump["analys_result_6_landmarks"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("analys_experiment_result_6_landmarks"):
        num = controller_dump["number_6_landmarks"]
        name = controller_dump["name_6_landmarks"]
        exp_points = {"all": "All", 1: "r", 2: "l", 3: "n"}
        cases_zero = {cn for cn, cd in dict_all_case.items() if cd.get("BR - plane offset", 0) == 0}
        cases_nonzero = {cn for cn, cd in dict_all_case.items() if cd.get("BR - plane offset", 0) != 0}
        landmarks_analysis(Path(data_path), dict_all_case,
                           ds_folder_name=f"Dataset{num}_{name}",
                           find_center_mass=True, probabilities_map=True,
                           active_point_name=exp_points,
                           case_subset=cases_zero,
                           group_label="_zero_offset",
                           result_folder_name=f"Exp_Dataset{num}_{name}")
        landmarks_analysis(Path(data_path), dict_all_case,
                           ds_folder_name=f"Dataset{num}_{name}",
                           find_center_mass=True, probabilities_map=True,
                           active_point_name=exp_points,
                           case_subset=cases_nonzero,
                           group_label="_nonzero_offset",
                           result_folder_name=f"Exp_Dataset{num}_{name}")
        controller_dump["analys_experiment_result_6_landmarks"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("analys_result_gh_landmark"):
        num = controller_dump["number_gh_landmark"]
        name = controller_dump["name_gh_landmark"]
        landmarks_analysis(Path(data_path), dict_all_case, ds_folder_name=f"Dataset{num}_{name}",
                           find_center_mass=True, probabilities_map=True, type_set="gh_landmark")
        controller_dump["analys_result_gh_landmark"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("analys_result_gh_lines"):
        num = controller_dump["number_gh_lines"]
        name = controller_dump["name_gh_lines"]
        predictions = curve_lines_analysis(data_path, result_folder, folder_name=f"Dataset{num}_{name}",
                                           dict_cases=dict_all_case, keys_to_need={1: 'RGH', 2: 'LGH', 3: 'NGH'},
                                           name_result_folder="geometric_height_comparison",
                                           spline_smoothing=0.5,
                                           probabilities_map=False, original_mask=True,
                                           points2points=False, curve2points=True,
                                           n_curve_points=20)
        if predictions:
            for case_name, pred_coords in predictions.items():
                if case_name in dict_all_case:
                    dict_all_case[case_name].update(pred_coords)
            json_save(dict_all_case, dict_all_case_path)
        controller_dump["analys_result_gh_lines"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("analys_result_ci_lines"):
        num = controller_dump["number_ci_lines"]
        name = controller_dump["name_ci_lines"]
        predictions = curve_lines_analysis(data_path, result_folder, folder_name=f"Dataset{num}_{name}",
                                           dict_cases=dict_all_case, keys_to_need={1: 'RCI', 2: 'LCI', 3: 'NCI'},
                                           name_result_folder="cusp_insertion_comparison",
                                           spline_smoothing=0.1,
                                           probabilities_map=False, original_mask=True,
                                           points2points=False, curve2points=True,
                                           n_curve_points=20)
        if predictions:
            for case_name, pred_coords in predictions.items():
                if case_name in dict_all_case:
                    dict_all_case[case_name].update(pred_coords)
            json_save(dict_all_case, dict_all_case_path)
        controller_dump["analys_result_ci_lines"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("calc_morphometric"):
        find_morphometric_parameters(data_path, ds_folder_name="Dataset412_SixAortaLandmarks")
        controller_dump["calc_morphometric"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("export_predicted_case_points"):
        template_path = os.path.join(script_dir, "templates", "case_point_template.txt")
        pred_output_folder = os.path.join(result_folder, "Predicted case points")
        clear_folder(pred_output_folder)

        with open(template_path, "r") as f:
            tmpl_lines = [line.rstrip("\n") for line in f]

        sections = []
        i = 1  # skip "case_name" header line
        while i < len(tmpl_lines):
            line = tmpl_lines[i].strip()
            if not line or line == "-":
                i += 1
                continue
            label = line
            dash_count = 0
            i += 1
            while i < len(tmpl_lines) and tmpl_lines[i].strip() == "-":
                dash_count += 1
                i += 1
            sections.append((label, dash_count >= 2))

        def _fmt_pt(coords):
            return " ".join(f"{v:.2f}" for v in coords)

        def _label_to_pred_key(tmpl_label):
            return tmpl_label.replace(" - closed", "") + "_pred"

        csv_rows = []
        section_labels = [s[0] for s in sections]

        for case_name, case_data in dict_all_case.items():
            if case_name not in test_cases:
                continue

            out_lines = [case_name, ""]
            csv_row = {"case_name": case_name}

            for label, is_multi in sections:
                pred_key = _label_to_pred_key(label)
                value = case_data.get(pred_key)
                out_lines.append(label)

                if value is None:
                    csv_row[label] = "x"
                    out_lines.append("-")
                    if is_multi:
                        out_lines.append("-")
                else:
                    csv_row[label] = "o"
                    if is_multi:
                        for pt in value:
                            out_lines.append(_fmt_pt(pt))
                    else:
                        out_lines.append(_fmt_pt(value))
                out_lines.append("")

            out_path = os.path.join(pred_output_folder, f"{case_name}.txt")
            with open(out_path, "w") as f:
                f.write("\n".join(out_lines))
            csv_rows.append(csv_row)

        df_pred = pd.DataFrame(csv_rows, columns=["case_name"] + section_labels)
        df_pred.to_csv(os.path.join(pred_output_folder, "predicted_case_points_presence.csv"), index=False)

        controller_dump["export_predicted_case_points"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump["experiment"]:
        experiment_training(create_img=False, create_models=True)
        experiment_analysis(data_path=data_path,
                            dict_case = {10: 481, 9: 489, 8: 488, 7: 487, 6: 486, 5: 485, 4: 484})

    print('hi')


if __name__ == "__main__":
    current_os = platform.system()
    script_dir = Path(__file__).resolve().parent

    if current_os == "Windows":
        data_path = "C:/Users/Kamil/Aortic_valve/data/"
        # data_path = "C:/Users/Kamil/Aortic_valve/data/temp"
        # data_path = "D:/science/Aortic_valve/data/"
    elif current_os == "Linux":
        data_path = "/home/kamili/aortic_valve/data/"
    else:
        data_path = None

    script_dir = Path(__file__).resolve().parent
    # data_structure_path = os.path.join(script_dir, "dir_structure.json")
    # dir_structure = json_reader(data_structure_path)
    # create_directory_structure(data_path, json_reader(data_structure_path))

    if data_path:
        _cleanup_empty_logs(os.path.join(data_path, "result/log"))
        free_cpus = get_free_cpus()
        current_time = datetime.now()

        # Логгер для хода программы
        log_file_name = current_time.strftime("log_%d-%m-%Y__%H-%M.log")
        work_log_path = os.path.join(data_path, "result/log", log_file_name)
        work_logger = logging.getLogger("work_logger")
        work_logger.setLevel(logging.INFO)
        work_handler = logging.FileHandler(work_log_path, mode='w')
        work_handler.setFormatter(logging.Formatter('%(message)s'))
        work_logger.addHandler(work_handler)

        # Логгер для результатов
        result_file_name = current_time.strftime("result_%d-%m-%Y__%H-%M.log")
        result_log_path = os.path.join(data_path, "result/log", result_file_name)
        result_logger = logging.getLogger("result_logger")
        result_logger.setLevel(logging.INFO)
        result_handler = logging.FileHandler(result_log_path, mode='w')
        result_handler.setFormatter(logging.Formatter('%(message)s'))  # без времени
        result_logger.addHandler(result_handler)

        # # Очистим базовые хендлеры у root-логгера (если есть)
        # for handler in logging.root.handlers[:]:
        #     logging.root.removeHandler(handler)

        controller(data_path, cpus=free_cpus)

