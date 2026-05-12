import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os


def _save_combined_mean_sd(df, measurements, save_txt_path,
                           method1_col="abs_err_gnn", method2_col="abs_err_center",
                           method1_label="GNN", method2_label="Center"):
    """
    Считает mean ± sd для объединённого списка measurements
    и сохраняет результат в txt.
    """

    # фильтрация по списку measurement
    subset = df[df["measurement"].isin(measurements)]

    if subset.empty:
        raise ValueError("Нет данных для переданных measurement.")

    m1_vals = subset[method1_col].dropna()
    m2_vals = subset[method2_col].dropna()

    if len(m1_vals) == 0 and len(m2_vals) == 0:
        raise ValueError("Нет числовых значений для расчёта.")

    m1_mean = m1_vals.mean()
    m1_sd = m1_vals.std()

    m2_mean = m2_vals.mean()
    m2_sd = m2_vals.std()

    # --- запись в txt ---
    with open(save_txt_path, "w", encoding="utf-8") as f:

        f.write("Combined measurements\n")
        f.write(f"Number of rows: {len(subset)}\n\n")

        f.write(f"{method1_label} (mean ± sd): ")
        f.write(f"{m1_mean:.3f} ± {m1_sd:.3f}\n")

        f.write(f"{method2_label} (mean ± sd): ")
        f.write(f"{m2_mean:.3f} ± {m2_sd:.3f}\n")


def _create_summary_table_plot(df, parameter_keys, save_path, save_txt_path, title,
                               method1_col="abs_err_gnn", method2_col="abs_err_center",
                               method1_label="GNN", method2_label="Center of mass"):

    rows = []

    for m in parameter_keys:
        subset = df[df["measurement"] == m]

        if subset.empty:
            continue

        m1_vals = subset[method1_col].dropna()
        m2_vals = subset[method2_col].dropna()

        if len(m1_vals) == 0 and len(m2_vals) == 0:
            continue

        m1_mean = m1_vals.mean()
        m1_sd = m1_vals.std()

        m2_mean = m2_vals.mean()
        m2_sd = m2_vals.std()

        label = parameter_keys[m][0]
        full_name = parameter_keys[m][1]

        rows.append([
            label,
            full_name,
            f"{m1_mean:.2f} ± {m1_sd:.2f}",
            f"{m2_mean:.2f} ± {m2_sd:.2f}"
        ])

    # --- создаём figure ---
    fig, ax = plt.subplots(figsize=(7, 0.21 * len(rows) + 0.5))
    ax.axis("off")

    col_label_1 = f"{method1_label}\n(mean ± sd)"
    col_label_2 = f"{method2_label}\n(mean ± sd)"

    table = ax.table(
        cellText=rows,
        colLabels=["Label", "Measurement", col_label_1, col_label_2],
        loc="center",
        #cellLoc="left"
        bbox=[0, 0, 1, 1]  # растянуть на всю область
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(4)))

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(ha='center')
        else:
            cell.set_text_props(ha='left')
            cell.PAD = 0.05

    # увеличить высоту строки заголовков
    for col in range(4):
        header_cell = table[(0, col)]
        header_cell.set_height(header_cell.get_height() * 1.8)

    # plt.title(title, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    # --- сохранение в txt ---
    with open(save_txt_path, "w", encoding="utf-8") as f:

        # заголовки
        headers = ["Label", "Measurement", f"{method1_label} (mean ± sd)", f"{method2_label} (mean ± sd)"]
        f.write("\t".join(headers) + "\n")

        # строки
        for row in rows:
            f.write("\t".join(row) + "\n")


def _create_aggregated_norm_table(df, groups, save_path, save_txt_path,
                                   method1_col="abs_err_gnn", method2_col="abs_err_center",
                                   method1_label="GNN", method2_label="Center of mass"):
    """
    Aggregated normalized MAE table.

    For each measurement: norm_MAE = mean(abs_err) / mean(ref) * 100 %
    For each group: report mean ± std of per-measurement norm_MAEs.

    groups: dict {group_label: [measurement_name, ...]}
    """
    rows = []

    for group_name, measurements in groups.items():
        norm_m1, norm_m2 = [], []

        for m in measurements:
            subset = df[df["measurement"] == m]
            if subset.empty:
                continue
            mean_ref = subset["ref"].dropna().mean()
            if not np.isfinite(mean_ref) or mean_ref == 0:
                continue
            mean_err1 = subset[method1_col].dropna().mean()
            mean_err2 = subset[method2_col].dropna().mean()
            if np.isfinite(mean_err1):
                norm_m1.append(mean_err1 / mean_ref * 100.0)
            if np.isfinite(mean_err2):
                norm_m2.append(mean_err2 / mean_ref * 100.0)

        def _fmt(vals):
            if not vals:
                return "—"
            mean = np.mean(vals)
            std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            return f"{mean:.2f} ± {std:.2f}%"

        rows.append([group_name, _fmt(norm_m1), _fmt(norm_m2)])

    if not rows:
        return

    # --- figure ---
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(rows) + 0.8))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=["Group", f"{method1_label}\nnorm. MAE (%)", f"{method2_label}\nnorm. MAE (%)"],
        loc="center",
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(3)))

    for (row, col), cell in table.get_celld().items():
        cell.set_text_props(ha='center' if (row == 0 or col > 0) else 'left')
        if row > 0:
            cell.PAD = 0.06
    for col in range(3):
        table[(0, col)].set_height(table[(0, col)].get_height() * 1.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    # --- txt ---
    with open(save_txt_path, "w", encoding="utf-8") as f:
        headers = ["Group", f"{method1_label} norm. MAE (%)", f"{method2_label} norm. MAE (%)"]
        f.write("\t".join(headers) + "\n")
        for row in rows:
            f.write("\t".join(row) + "\n")


def _plot_group(df, parameter_keys, save_path, title,
                method1_col="abs_err_gnn", method2_col="abs_err_center",
                method1_label="GNN", method2_label="Center of mass"):
    data = []
    positions = []
    labels = []
    group_centers = []
    separator_positions = []

    pos = 1
    box_widths = 0.2 # ширина бара
    pair_spacing = 0.3  # расстояние внутри пары
    group_spacing = 0.4  # расстояние между группами
    width_per_group = 0.6  # ширина на одну пару (в дюймах)
    base_margin = 1.2  # боковые поля
    fixed_height = 5

    for m in parameter_keys.keys():
        labels.append(parameter_keys[m][0])  # берём label
        subset = df[df["measurement"] == m]

        if subset.empty:
            continue

        m1_vals = subset[method1_col].dropna().values
        m2_vals = subset[method2_col].dropna().values

        if len(m1_vals) == 0 and len(m2_vals) == 0:
            continue

        data.append(m2_vals)
        data.append(m1_vals)

        positions.append(pos)
        positions.append(pos + pair_spacing)

        # центр группы (между двумя боксами)
        group_centers.append(pos + pair_spacing / 2)

        separator_positions.append(pos + pair_spacing + group_spacing / 2)

        pos += pair_spacing + group_spacing

    fig_width = len(labels) * width_per_group + base_margin
    fig, ax = plt.subplots(figsize=(fig_width, fixed_height))

    box = ax.boxplot(
        data,
        positions=positions,
        widths=box_widths,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        showfliers=False,
        meanprops=dict(color="black", linewidth=1.5),
        medianprops = dict(color="none")
    )

    # --- покраска ---
    for i, patch in enumerate(box["boxes"]):
        if i % 2 == 0:
            patch.set_facecolor("#DD8452")  # method2
        else:
            patch.set_facecolor("#4C72B0")  # method1

    # --- разделительные линии ---
    for sep in separator_positions[:-1]:
        ax.axvline(sep, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

    # --- легенда ---
    legend_elements = [
        Patch(facecolor="#DD8452", label=method2_label),
        Patch(facecolor="#4C72B0", label=method1_label)
    ]
    ax.legend(handles=legend_elements)

    # --- подписи оси X ---
    ax.set_xticks(group_centers)
    ax.set_xticklabels(labels=labels)#, rotation=45, ha="right")

    ax.set_ylabel(title)
    # ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def creat_box_plot(result_path, file_name,
                   method1_col="abs_err_gnn", method2_col="abs_err_center",
                   method1_label="GNN", method2_label="Center of mass"):
    data_file_path = os.path.join(result_path, file_name)
    plot_length_img_path = os.path.join(result_path, 'boxplot_length.png')
    plot_angle_img_path = os.path.join(result_path, 'boxplot_angle.png')
    length_table_path = os.path.join(result_path, "table_length.png")
    angle_table_path = os.path.join(result_path, "table_angle.png")
    length_txt_path = os.path.join(result_path, "table_length.txt")
    angle_txt_path = os.path.join(result_path, "table_angle.txt")
    angle_mean_txt_path = os.path.join(result_path, "mean_angle.txt")
    df = pd.read_csv(str(data_file_path))

    # measurements = sorted(df["measurement"].unique())

    length_par_dict ={
        'BR_perimeter': ["BR_per", "Basal ring perimeter"],
        'BR_max': ["BR_max", "Maximum basal ring diameter"],
        'BR_min': ["BR_min", "Minimum basal ring diameter"],
        'BR_diameter': ["BR_avg", "Mean basal ring diameter"],
        'IC_R': ["IC_R","Right intercommissural distance"],
        'IC_L': ["IC_L", "Left intercommissural distance"],
        'IC_N': ["IC_N", "Non-coronary intercommissural distance"],
        'IC_distance': ["IC_avg", "Mean intercommissural distance"],
        'RL_comm_height': ["CH_RL", "Commissural height between right and left leaflets"],
        'RN_comm_height': ["CH_RN", "Commissural height between right and non-coronary leaflets"],
        'LN_comm_height': ["CH_LN", "Commissural height between left and non-coronary leaflets"],
        'mean_comm_heigh': ["CH_avg", "Mean commissural height"],
        'ST_perimeter': ["ST_per", "Sino-tubular junction perimeter"],
        'ST_max': ["ST_max", "Maximum STJ diameter"],
        'ST_min': ["ST_min", "Minimum STJ diameter"],
        'ST_diameter': ["ST_avg", "Mean STJ diameter"],
        'commissural_diameter': ["CD", "Commissural diameter"],
        'centroid_valve_height': ["CVH", "Centroid valve height"]
    }

    angle_par_dict = {
        'RL_angle': ["LA_RL", "Angle between right and left leaflets"],
        'RN_angle': ["LA_RN", "Angle between right and non-coronary leaflets"],
        'LN_angle': ["LA_LN", "Angle between left and non-coronary leaflets"],
        'R_flat_angle': ["FA_R", "Right leaflet flat angle"],
        'L_flat_angle': ["FA_L", "Left leaflet flat angle"],
        'N_flat_angle': ["FA_N", "Non-coronary leaflet flat angle"],
        'R_vertical_angle': ["VA_R", "Right leaflet vertical angle"],
        'L_vertical_angle': ["VA_L", "Left leaflet vertical angle"],
        'N_vertical_angle': ["VA_N", "Non-coronary leaflet vertical angle"],
        'mean_vertical_angle': ["VA_avg", "Mean vertical leaflet angle"],
        'BR_C_plane_angle': ["BCA", "Angle between the basal ring plane and the commissural plane"]
    }
    # 'mean_commissural_angle': "Mean commissural angle",

    angle_list = [
        'R_flat_angle', 'L_flat_angle', 'N_flat_angle', 'R_vertical_angle', 'L_vertical_angle', 'N_vertical_angle',
        'RL_angle', 'RN_angle', 'LN_angle', 'BR_C_plane_angle'
    ]

    aggregated_table_path = os.path.join(result_path, "table_aggregated.png")
    aggregated_txt_path = os.path.join(result_path, "table_aggregated.txt")

    _save_combined_mean_sd(df, angle_list, angle_mean_txt_path,
                           method1_col, method2_col, method1_label, method2_label)

    _plot_group(df, length_par_dict, plot_length_img_path, "MAE, mm",
                method1_col, method2_col, method1_label, method2_label)

    _plot_group(df, angle_par_dict, plot_angle_img_path, "MAE, °",
                method1_col, method2_col, method1_label, method2_label)

    _create_summary_table_plot(df, length_par_dict,
                               length_table_path, length_txt_path, "Length Measurements",
                               method1_col, method2_col, method1_label, method2_label)

    _create_summary_table_plot(df, angle_par_dict,
                               angle_table_path, angle_txt_path, "Angle Measurements",
                               method1_col, method2_col, method1_label, method2_label)

    aggregated_groups = {
        "Length measurements": list(length_par_dict.keys()),
        "Angle measurements": list(angle_par_dict.keys()),
    }
    _create_aggregated_norm_table(df, aggregated_groups,
                                   aggregated_table_path, aggregated_txt_path,
                                   method1_col, method2_col, method1_label, method2_label)

if __name__ == "__main__":

    result_path = r'C:\Users\Kamil\Aortic_valve\data\gnn_folder\results'
    # result_path = r'D:\science\Aortic_valve\GNN\results'
    file_name = "gnn_vs_center_comparison.csv"
    creat_box_plot(result_path, file_name)
