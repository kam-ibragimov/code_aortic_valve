import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from data_postprocessing.evaluation_analysis import  evaluate_segmentation
from data_preprocessing.text_worker import add_info_logging
from skimage.morphology import skeletonize_3d
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter, label, generate_binary_structure
from skimage.filters import threshold_otsu


def load_new_coords_org(mask_path, label, coord_org, original_mask):
    if original_mask:
        org_mask = sitk.ReadImage(str(mask_path))
        org_map_all = sitk.GetArrayFromImage(org_mask)
        binary_mask_org = (org_map_all == label).astype(np.uint8)
        return new_spline_from_pixel_coord(binary_mask_org, str(mask_path))
    else:
        return new_spline_from_world_coord(coord_org)


def new_spline_from_world_coord(points, n_samples=10, smoothing=0.0):
    """
    points: np.ndarray (N,3) — точки в мировых координатах
    n_samples: количество равномерно распределённых точек по длине сплайна
    smoothing: параметр сглаживания (0 — проходит через все точки)
    """
    points = np.asarray(points, dtype=float)
    if points.shape[0] < 2:
        raise ValueError("Нужно хотя бы две точки")

    # параметризация по длине дуги
    d = np.linalg.norm(np.diff(points, axis=0), axis=1)
    t = np.concatenate([[0], np.cumsum(d)])
    t /= t[-1]  # нормализация в [0,1]

    # строим сплайн
    tck, _ = splprep(points.T, u=t, s=smoothing, k=min(3, len(points)-1))

    # равномерные точки по длине дуги
    u_new = np.linspace(0, 1, n_samples)
    new_points = np.array(splev(u_new, tck)).T
    return new_points

def new_spline_from_pixel_coord(mask, nii_path, n_samples=10, smoothing=10.0):
    def _voxel_to_world_batch(indices, image):
        """
        Преобразует индексы пикселей (i,j,k) в мировые координаты (x,y,z).
        """
        origin = np.array(image.GetOrigin())
        spacing = np.array(image.GetSpacing())
        direction = np.array(image.GetDirection()).reshape(3, 3)
        indices = np.asarray(indices, dtype=float)
        world = (indices * spacing) @ direction.T + origin
        return world

    """
    points: np.ndarray (N,3) — точки в мировых координатах
    n_samples: количество точек для равномерного сэмплирования
    """
    mask = np.swapaxes(mask, 0, -1)
    pixel_points = extract_centerline_from_heatmap(mask)
    image = sitk.ReadImage(nii_path)
    points = _voxel_to_world_batch(pixel_points, image)

    if points.shape[0] < 2:
        raise ValueError("Нужно хотя бы две точки")

    d = np.linalg.norm(np.diff(points, axis=0), axis=1)
    t = np.concatenate([[0], np.cumsum(d)])
    t /= t[-1]

    tck, _ = splprep(points.T, u=t, s=smoothing, k=3)
    u_new = np.linspace(0, 1, n_samples)
    new_points = np.array(splev(u_new, tck)).T
    return new_points


def extract_centerline_from_heatmap(
        heatmap,  # np.ndarray (Z,Y,X), values ~[0,1]
        sigma=0.8,  # gaussian sigma in voxels
        thresh=None,  # if None, compute candidate thresholds automatically
        perc=90,  # percentile used for fallback threshold selection
):
    assert heatmap.ndim == 3

    # 1. сглаживание
    sm = gaussian_filter(heatmap.astype(np.float32), sigma=sigma)

    # 2. варианты порога
    if thresh is None:
        # 2a. Otsu (вдруг распределение бимодальное)
        try:
            flat = sm.ravel()
            # Otsu может быть плох на сильно несбалансированных данных, но попробуем
            flat = flat[flat > np.percentile(flat, 99)]  # отсечь нижние %
            otsu_t = threshold_otsu(flat)
        except Exception:
            otsu_t = None

        # 2b. percentile (80-й, например)
        perc_t = float(np.percentile(sm, perc))

        # 2c. fallback heuristics: ограничим percentile в разумных пределах
        # (если perc_t слишком велик — возможно шум; ограничим 0.05..0.6)
        perc_t = float(np.clip(perc_t, 0.01, 0.8))

        # choose threshold: если Otsu в разумных границах, используем его, иначе percentile
        if otsu_t is not None and otsu_t > perc_t:
            chosen = float(otsu_t)
        else:
            chosen = perc_t
    else:
        chosen = float(thresh)

    # chosen = 0.9

    # 3. бинаризация
    binary = sm > chosen

    # 4. компоненты (для быстрой диагностики)
    labeled, ncomp = label(binary)

    if ncomp > 0:
        # вычисляем размер каждой компоненты
        sizes = np.array([np.sum(labeled == lab) for lab in range(1, ncomp + 1)])
        # находим индекс самой большой (нумерация с 1)
        largest_label = np.argmax(sizes) + 1
        # создаем маску только крупнейшей компоненты
        mask_largest = (labeled == largest_label)
    else:
        # если ничего не найдено, создаём пустую маску
        mask_largest = np.zeros_like(sm, dtype=bool)

    sm_largest = sm * mask_largest

    skeleton = skeletonize_3d(mask_largest)

    structure = generate_binary_structure(rank=3, connectivity=3)
    labeled, ncomp = label(skeleton, structure=structure)
    if ncomp == 0:
        return np.zeros_like(skeleton, dtype=bool), np.empty((0, 3), dtype=int)

    # Считаем длины всех ветвей
    lengths = np.array([np.sum(labeled == i) for i in range(1, ncomp + 1)])

    # Выбираем самую длинную ветвь
    main_label = np.argmax(lengths) + 1

    # Оставляем только её
    centerline = (labeled == main_label)

    coords = np.argwhere(centerline)

    return coords


def extract_boundary_curve_world(mask_array, mask_sitk):
    """
    Extracts a face-connected closed boundary contour from an oblique 2D mask
    stored as a single-slice 3D nii.gz (one dimension == 1, any axis).
    Returns np.ndarray of shape (N, 3) in world coordinates (mm).
    """
    from skimage.measure import find_contours

    origin = np.array(mask_sitk.GetOrigin())
    spacing = np.array(mask_sitk.GetSpacing())
    direction = np.array(mask_sitk.GetDirection()).reshape(3, 3)

    # numpy axes are (Z, Y, X) in ITK terms: numpy axis k → ITK axis (2-k)
    # Find which numpy axis has size 1 (the plane normal axis)
    thin_numpy = int(np.argmin(mask_array.shape))
    plane_numpy = [i for i in range(3) if i != thin_numpy]

    slice_2d = (np.squeeze(mask_array, axis=thin_numpy) > 0).astype(np.uint8)
    contours = find_contours(slice_2d, level=0.5)
    if not contours:
        return np.empty((0, 3))

    contour = max(contours, key=len)
    coords = np.array(contour)  # (N, 2): [row, col] in the 2D slice

    # Map (row, col) back to 3D ITK pixel indices.
    # After squeezing thin_numpy, the 2D axes are plane_numpy in ascending order:
    #   2D axis 0 (row) → numpy axis plane_numpy[0] → ITK axis (2 - plane_numpy[0])
    #   2D axis 1 (col) → numpy axis plane_numpy[1] → ITK axis (2 - plane_numpy[1])
    idx_3d = np.zeros((len(coords), 3))
    idx_3d[:, 2 - plane_numpy[0]] = coords[:, 0]  # row
    idx_3d[:, 2 - plane_numpy[1]] = coords[:, 1]  # col
    # thin ITK axis index stays 0

    world_points = (idx_3d * spacing) @ direction.T + origin
    return world_points


def mask_comparison(data_path, type_mask, folder_name):
    nnUNet_folder = os.path.join(data_path, "nnUNet_folder")
    result_mask_folder = os.path.join(nnUNet_folder, "nnUNet_test", folder_name)
    original_mask_folder = os.path.join(nnUNet_folder, "original_mask", folder_name)

    per_case_data = []

    for case in os.listdir(result_mask_folder):
        if not case.endswith(".nii.gz"):
            continue
        case_name = case[:-7]
        first_char = case_name[0]

        result_mask_path = os.path.join(result_mask_folder, case)
        original_mask_path = os.path.join(original_mask_folder, f"{case_name}.nii.gz")

        if not os.path.exists(original_mask_path):
            add_info_logging(f"Missing {case_name} - no original mask", "work_logger")
            continue

        result_mask = nib.load(result_mask_path).get_fdata()
        mask_img = nib.load(original_mask_path).get_fdata()

        if result_mask.shape != mask_img.shape:
            add_info_logging(f"The dimensions do not match for the case: {case_name}", "work_logger")
            add_info_logging(f"result_mask shape:  {result_mask.shape}", "work_logger")
            add_info_logging(f"original_mask shape:{mask_img.shape}", "work_logger")
            continue  # пропустить

        try:
            metrics = evaluate_segmentation(mask_img, result_mask)
            # Сохраняем по кейсу
            metrics["case"] = case_name
            metrics["group"] = first_char
            per_case_data.append(metrics)
        except Exception as e:
            add_info_logging(f"Error while comparing {case_name}: {str(e)}", "work_logger")

    return per_case_data


class LandmarkCentersCalculator:

    @staticmethod
    def _compute_center_of_mass(binary_mask, spacing, origin, direction):
        # Function to compute center of mass in world coordinates
        indices = np.argwhere(binary_mask)  # Get voxel indices of the mask
        if len(indices) == 0:
            return None  # No center of mass if mask is empty

        # Compute the mean position in voxel space
        center_voxel = np.mean(indices, axis=0)[::-1]  # Reverse order (Z, Y, X) -> (X, Y, Z)

        # Convert to world coordinates using the corrected direction matrix
        center_world = np.dot(direction, center_voxel * spacing) + origin
        return center_world

    def extract_landmarks_com_nii(self, mask_nii):
        mask_image = sitk.ReadImage(mask_nii)
        mask_array = sitk.GetArrayFromImage(mask_image)  # Convert to NumPy array

        # Get image metadata
        spacing = np.array(mask_image.GetSpacing())  # (x, y, z) voxel size
        origin = np.array(mask_image.GetOrigin())  # World coordinate of (0,0,0)
        direction = np.array(mask_image.GetDirection()).reshape(3, 3)  # Reshape to 3x3 matrix

        # Find unique labels (excluding background 0)
        labels = np.unique(mask_array)
        labels = labels[labels != 0]  # Remove background if label 0 exists

        # Compute center of mass for each label
        centers_of_mass = {}
        for label in labels:
            binary_mask = (mask_array == label)  # Create binary mask for current label
            center_world = self._compute_center_of_mass(binary_mask, spacing, origin, direction)
            if center_world is not None:
                centers_of_mass[label] = center_world

        return centers_of_mass

    def extract_landmarks_com_npz(self, mask_nii, mask_npz):
        # Get image metadata
        mask_image = sitk.ReadImage(mask_nii)
        spacing = np.array(mask_image.GetSpacing())  # (x, y, z) voxel size
        origin = np.array(mask_image.GetOrigin())  # World coordinate of (0,0,0)
        direction = np.array(mask_image.GetDirection()).reshape(3, 3)  # Reshape to 3x3 matrix

        prob_map_all = np.load(mask_npz)
        labels = len(prob_map_all["probabilities"])

        # Compute center of mass for each label
        centers_of_mass = {}
        for label in range(1, labels):
            binary_mask = prob_map_all["probabilities"][label]  # Create binary mask for current label
            binary_mask[binary_mask < np.max(binary_mask)*0.2] = 0
            center_world = self._compute_center_of_mass(binary_mask, spacing, origin, direction)
            if center_world is not None:
                centers_of_mass[label] = center_world

        return centers_of_mass

    @staticmethod
    def extract_landmarks_peak_npz(mask_nii, mask_npz):
        # Get image metadata
        mask_image = sitk.ReadImage(mask_nii)
        spacing = np.array(mask_image.GetSpacing())  # (x, y, z)
        origin = np.array(mask_image.GetOrigin())  # (x0, y0, z0)
        direction = np.array(mask_image.GetDirection()).reshape(3, 3)

        prob_map_all = np.load(mask_npz)
        labels = len(prob_map_all["probabilities"])

        # Compute peak voxel for each label
        peaks_world = {}
        for label in range(1, labels):
            prob_map = prob_map_all["probabilities"][label]
            if np.max(prob_map) == 0:
                continue  # Skip empty maps

            # Find voxel index of the maximum probability
            peak_index = np.unravel_index(np.argmax(prob_map), prob_map.shape)  # (z, y, x)

            # Convert index to physical coordinates
            peak_voxel = np.array(peak_index)[::-1]  # Convert to (x, y, z) from (z, y, x)
            peak_world = np.dot(direction, peak_voxel * spacing) + origin

            peaks_world[label] = peak_world

        return peaks_world

    @staticmethod
    def extract_landmarks_topk_peaks_npz(mask_nii, mask_npz, top_k=3):
        # Get image metadata
        mask_image = sitk.ReadImage(mask_nii)
        spacing = np.array(mask_image.GetSpacing())  # (x, y, z)
        origin = np.array(mask_image.GetOrigin())  # (x0, y0, z0)
        direction = np.array(mask_image.GetDirection()).reshape(3, 3)

        prob_map_all = np.load(mask_npz)
        labels = len(prob_map_all["probabilities"])

        # Compute top-k peak voxels for each label
        peaks_world = {}
        for label in range(1, labels):
            prob_map = prob_map_all["probabilities"][label]
            flat = prob_map.flatten()
            if np.max(flat) == 0:
                continue  # Skip empty maps

            # Get indices of top_k highest probabilities
            topk_indices_flat = np.argpartition(-flat, range(min(top_k, flat.size)))[:top_k]
            topk_indices_sorted = topk_indices_flat[np.argsort(-flat[topk_indices_flat])]

            peak_world_coords = []
            for idx in topk_indices_sorted:
                peak_index = np.unravel_index(idx, prob_map.shape)  # (z, y, x)
                peak_voxel = np.array(peak_index)[::-1]  # to (x, y, z)
                peak_world = np.dot(direction, peak_voxel * spacing) + origin
                peak_world_coords.append(peak_world)

            peaks_world[label] = peak_world_coords

        return peaks_world  # Dict[label] = [coord1, coord2, ..., coordK]
