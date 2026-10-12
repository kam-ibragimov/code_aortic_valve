import os
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label

from data_preprocessing.text_worker import add_info_logging
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from models.implementation_nnUnet import nnUnet_trainer


def _clear_folder(folder):
    """Очищает папку, удаляя все файлы и подпапки"""
    if not folder.exists():
        add_info_logging(f"Folder '{str(folder)}' does not exist.", "work_logger")
        return

    for item in folder.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()  # Удаляем файл или символическую ссылку
        elif item.is_dir():
            shutil.rmtree(item)  # Удаляем папку рекурсивно


def _configure_folder(base_folder, ds_folder_name):
    subfolders = ["imagesTr", "imagesTs", "labelsTr"]

    (base_folder/"nnUNet_preprocessed").mkdir(exist_ok=True)
    _clear_folder(base_folder/"nnUNet_raw"/ds_folder_name)
    (base_folder/"nnUNet_raw"/ds_folder_name).mkdir(parents=True, exist_ok=True)
    (base_folder/"nnUNet_results").mkdir(exist_ok=True)
    (base_folder/"nnUNet_test").mkdir(exist_ok=True)
    _clear_folder(base_folder/"original_mask"/ds_folder_name)
    (base_folder/"original_mask"/ds_folder_name).mkdir(parents=True, exist_ok=True)

    # Создаем папки, если их нет
    for subfolder in subfolders:
        (base_folder/"nnUNet_raw"/ds_folder_name/subfolder).mkdir(exist_ok=True)


def _get_file_list(df, series_type, column, suffix, base_path):
    return [
        base_path / f"{name}{suffix}"
        for name in df[df["type_series"] == series_type][column].dropna()
    ]


def _copy_img(input_imgs_path, output_folder, rename=False):
    _clear_folder(output_folder)
    for img_path in input_imgs_path:
        if rename:
            case_name = img_path.name[:-7]
            shutil.copy(img_path, output_folder/f"{case_name}_0000.nii.gz")
        else:
            shutil.copy(img_path, output_folder/img_path.name)


def _keep_largest_component(mask_path):
    """
    Оставляет в маске только наибольший связный компонент переднего плана (все метки > 0 вместе),
    остальные компоненты обнуляет и перезаписывает файл. Связность 26 соседей — как в nnU-Net
    (remove_all_but_largest_component). Компоненты ищутся внутри bounding box переднего плана:
    результат тот же, но не нужен массив меток на весь объём (на больших КТ это десятки ГБ).
    Возвращает число удалённых вокселей.
    """
    image = sitk.ReadImage(str(mask_path))
    seg = sitk.GetArrayFromImage(image)
    foreground = seg > 0
    if not foreground.any():
        return 0

    slices = []
    for axis in range(foreground.ndim):
        other = tuple(i for i in range(foreground.ndim) if i != axis)
        idx = np.flatnonzero(foreground.any(axis=other))
        slices.append(slice(idx[0], idx[-1] + 1))
    slices = tuple(slices)

    labeled, num = label(foreground[slices], structure=np.ones((3,) * foreground.ndim, dtype=bool))
    if num <= 1:
        return 0
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    remove = (labeled != 0) & (labeled != np.argmax(sizes))
    removed = int(np.count_nonzero(remove))

    seg_crop = seg[slices]
    seg_crop[remove] = 0

    new_image = sitk.GetImageFromArray(seg)
    new_image.CopyInformation(image)
    sitk.WriteImage(new_image, str(mask_path), useCompression=True)
    return removed


def process_nnunet(folder, ds_folder_name, id_case, folder_image_path,
                   folder_mask_path, dict_dataset, train_test_lists, create_ds=False,
                   training_mod=False, predicting_mod=False, save_probabilities=True,
                   network="3d_fullres", keep_largest_component=False):

    folder = Path(folder)

    if create_ds:
        paths_to_check = [
            folder / "nnUNet_raw" / ds_folder_name,
            folder / "original_mask" / ds_folder_name,
        ]

        for p in paths_to_check:
            if p.exists():
                add_info_logging(f"Folder already exists: {p}. Please choose a different name ds_folder_name.")
                raise FileExistsError(f"Folder already exists: {p}. Please choose a different name ds_folder_name.")

        folder_image_path = Path(folder_image_path)
        folder_mask_path = Path(folder_mask_path)
        _configure_folder(folder, ds_folder_name)

        list_train_case = _get_file_list(train_test_lists, "train", "used_case_name", ".nii.gz", folder_image_path)
        list_train_mask = _get_file_list(train_test_lists, "train", "used_case_name", ".nii.gz", folder_mask_path)
        list_test_case = _get_file_list(train_test_lists, "test", "used_case_name", ".nii.gz", folder_image_path)
        list_test_mask = _get_file_list(train_test_lists, "test", "used_case_name", ".nii.gz", folder_mask_path)

        _copy_img(list_train_case, folder/ "nnUNet_raw" / ds_folder_name / "imagesTr", rename=True)
        _copy_img(list_train_mask, folder / "nnUNet_raw" / ds_folder_name / "labelsTr")
        _copy_img(list_test_case, folder/ "nnUNet_raw" / ds_folder_name / "imagesTs", rename=True)
        _copy_img(list_test_mask, folder / "original_mask" / ds_folder_name)

        generate_dataset_json(str(folder/ "nnUNet_raw" / ds_folder_name),
                              channel_names=dict_dataset["channel_names"],
                              labels=dict_dataset["labels"],
                              num_training_cases=len(list_train_case),
                              file_ending=dict_dataset["file_ending"])

    if training_mod:
        model_nnUnet = nnUnet_trainer(str(folder))
        model_nnUnet.preprocessing(task_id=id_case)
        model_nnUnet.train(task_id=id_case, fold="all", network=network)

    if predicting_mod:
        input_folder = Path(folder / "nnUNet_raw" / ds_folder_name / "imagesTs")
        output_folder = Path(folder / "nnUNet_test" / ds_folder_name)
        model_nnUnet = nnUnet_trainer(str(folder))
        model_nnUnet.predicting(input_folder=str(input_folder),
                                output_folder=str(output_folder),
                                task_id=id_case, fold="all",
                                save_probabilities=save_probabilities,
                                network=network)

        if keep_largest_component:
            # Постобработка: удаляем ложные фрагменты, маски перезаписываются на месте.
            # Файлы вероятностей (.npz) не меняются.
            for mask_path in sorted(output_folder.glob("*.nii.gz")):
                removed = _keep_largest_component(mask_path)
                if removed:
                    add_info_logging(f"{ds_folder_name}: {mask_path.name} - removed {removed} voxels "
                                     f"outside the largest component", "work_logger")
