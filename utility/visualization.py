import numpy as np
import nibabel as nib
from pathlib import Path
import SimpleITK as sitk
import json
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R


def controller(json_path):
    # === 1. Загрузка JSON ===
    with open(json_path, "r") as f:
        data = json.load(f)

    # === 2. Выделение ключевых точек для выравнивания ===
    p1 = np.array(data["R"])
    p2 = np.array(data["L"])
    p3 = np.array(data["N"])

    # === 3. Построение нормали к плоскости (через 3 точки) ===
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)

    # === 4. Вычисление поворота, чтобы normal → [0, 0, 1] (вверх) ===
    rotation = R.align_vectors([[0, 0, 1]], [normal])[0]

    def rotate_point(pt):
        return rotation.apply(pt)

    # === 5. Создание PyVista Plotter ===
    plotter = pv.Plotter()
    colors = pv.colors.hexcolors  # для выбора случайных цветов
    color_list = list(colors.values())

    # ключи, которые НЕ отрисовываются
    # skip_keys = ["RLS - closed", "RNS - closed", "LNS - closed", "BR - closed"]
    skip_keys = []

    custom_colors = {
        "RGH": "turquoise",
        "LGH": "turquoise",
        "NGH": "turquoise",
        "RCI": "red",
        "LCI": "red",
        "NCI": "red",
        "BR - closed": "black",
        "RLS - closed": "blue",
        "RNS - closed": "blue",
        "LNS - closed": "blue",
    }

    # === 6. Добавление всех кривых (дуг) ===
    for i, (key, val) in enumerate(data.items()):
        if key in skip_keys:
            continue
        if isinstance(val, list) and isinstance(val[0], list):  # кривая
            pts = np.array(val)
            pts_rot = rotate_point(pts)
            spline = pv.Spline(pts_rot, 200)
            # Цвет: из словаря или по умолчанию
            color = custom_colors.get(key, color_list[i % len(color_list)])
            plotter.add_mesh(spline, color=color, line_width=5, label=key)

    # === 7. Добавление одиночных точек и их подписей ===
    landmark_keys = ["R", "L", "N", "RLC", "RNC", "LNC"]
    for key in landmark_keys:
        pt = np.array(data[key])
        pt_rot = rotate_point(pt)
        plotter.add_mesh(pv.Sphere(radius=0.8, center=pt_rot), color="red")
        plotter.add_point_labels([pt_rot], [key], point_size=10, font_size=50, text_color="black", name=key)

    # === 8. Добавление легенды и отображение ===
    plotter.add_legend()
    plotter.show_grid()
    # plotter.show_bounds(xlabel=None, ylabel=None, zlabel=None)#, ticks=False)
    plotter.hide_axes()
    # plotter.remove_bounds_axes()
    plotter.show()
    print("hi")



if __name__ == "__main__":
    data_path = "C:/Users/Kamil/Aortic_valve/data/"
    json_path = Path(data_path) /"json_markers_info"/"Normal/n1.json"
    controller(json_path)