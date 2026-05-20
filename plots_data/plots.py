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
                               method1_label="GNN", method2_label="Center of mass",
                               method3_col=None, method3_label=None):

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

        row = [
            label,
            full_name,
            f"{m1_mean:.2f} ± {m1_sd:.2f}",
            f"{m2_mean:.2f} ± {m2_sd:.2f}",
        ]

        if method3_col is not None:
            m3_vals = subset[method3_col].dropna()
            m3_mean = m3_vals.mean()
            m3_sd = m3_vals.std()
            row.append(f"{m3_mean:.2f} ± {m3_sd:.2f}")

        rows.append(row)

    n_cols = 5 if method3_col is not None else 4
    fig_width = 7 if method3_col is None else 9

    fig, ax = plt.subplots(figsize=(fig_width, 0.21 * len(rows) + 0.5))
    ax.axis("off")

    col_label_1 = f"{method1_label}\n(mean ± sd)"
    col_label_2 = f"{method2_label}\n(mean ± sd)"
    col_labels = ["Label", "Measurement", col_label_1, col_label_2]
    if method3_col is not None:
        col_labels.append(f"{method3_label}\n(mean ± sd)")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        bbox=[0, 0, 1, 1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(n_cols)))

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(ha='center')
        else:
            cell.set_text_props(ha='left')
            cell.PAD = 0.05

    for col in range(n_cols):
        header_cell = table[(0, col)]
        header_cell.set_height(header_cell.get_height() * 1.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    with open(save_txt_path, "w", encoding="utf-8") as f:
        headers = ["Label", "Measurement", f"{method1_label} (mean ± sd)", f"{method2_label} (mean ± sd)"]
        if method3_col is not None:
            headers.append(f"{method3_label} (mean ± sd)")
        f.write("\t".join(headers) + "\n")

        for row in rows:
            f.write("\t".join(row) + "\n")


def _create_aggregated_norm_table(df, groups, save_path, save_txt_path,
                                   method1_col="abs_err_gnn", method2_col="abs_err_center",
                                   method1_label="GNN", method2_label="Center of mass",
                                   method3_col=None, method3_label=None):
    """
    Aggregated normalized MAE table.

    For each measurement: norm_MAE = mean(abs_err) / mean(ref) * 100 %
    For each group: report mean ± std of per-measurement norm_MAEs.

    groups: dict {group_label: [measurement_name, ...]}
    """
    rows = []

    def _fmt(vals):
        if not vals:
            return "—"
        mean = np.mean(vals)
        std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        return f"{mean:.2f} ± {std:.2f}%"

    for group_name, measurements in groups.items():
        norm_m1, norm_m2, norm_m3 = [], [], []

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
            if method3_col is not None and method3_col in df.columns:
                mean_err3 = subset[method3_col].dropna().mean()
                if np.isfinite(mean_err3):
                    norm_m3.append(mean_err3 / mean_ref * 100.0)

        row = [group_name, _fmt(norm_m1), _fmt(norm_m2)]
        if method3_col is not None:
            row.append(_fmt(norm_m3))
        rows.append(row)

    if not rows:
        return

    n_cols = 4 if method3_col is not None else 3
    fig_width = 7 if method3_col is None else 9

    fig, ax = plt.subplots(figsize=(fig_width, 0.4 * len(rows) + 0.8))
    ax.axis("off")

    col_labels = ["Group", f"{method1_label}\nnorm. MAE (%)", f"{method2_label}\nnorm. MAE (%)"]
    if method3_col is not None:
        col_labels.append(f"{method3_label}\nnorm. MAE (%)")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(n_cols)))

    for (row, col), cell in table.get_celld().items():
        cell.set_text_props(ha='center' if (row == 0 or col > 0) else 'left')
        if row > 0:
            cell.PAD = 0.06
    for col in range(n_cols):
        table[(0, col)].set_height(table[(0, col)].get_height() * 1.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    with open(save_txt_path, "w", encoding="utf-8") as f:
        headers = ["Group", f"{method1_label} norm. MAE (%)", f"{method2_label} norm. MAE (%)"]
        if method3_col is not None:
            headers.append(f"{method3_label} norm. MAE (%)")
        f.write("\t".join(headers) + "\n")
        for row in rows:
            f.write("\t".join(row) + "\n")


def _plot_group(df, parameter_keys, save_path, title,
                method1_col="abs_err_gnn", method2_col="abs_err_center",
                method1_label="GNN", method2_label="Center of mass",
                method3_col=None, method3_label=None):
    data = []
    positions = []
    labels = []
    group_centers = []
    separator_positions = []

    pos = 1
    box_widths = 0.2
    pair_spacing = 0.3
    group_spacing = 0.4
    width_per_group = 0.6
    base_margin = 1.2
    fixed_height = 5

    n_methods = 3 if method3_col is not None else 2

    for m in parameter_keys.keys():
        labels.append(parameter_keys[m][0])
        subset = df[df["measurement"] == m]

        if subset.empty:
            continue

        m1_vals = subset[method1_col].dropna().values
        m2_vals = subset[method2_col].dropna().values

        if len(m1_vals) == 0 and len(m2_vals) == 0:
            continue

        data.append(m2_vals)
        data.append(m1_vals)

        if n_methods == 3:
            m3_vals = subset[method3_col].dropna().values
            data.append(m3_vals)
            positions.append(pos)
            positions.append(pos + pair_spacing)
            positions.append(pos + 2 * pair_spacing)
            group_centers.append(pos + pair_spacing)
            separator_positions.append(pos + 2 * pair_spacing + group_spacing / 2)
            pos += 2 * pair_spacing + group_spacing
        else:
            positions.append(pos)
            positions.append(pos + pair_spacing)
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
        medianprops=dict(color="none")
    )

    colors = ["#DD8452", "#4C72B0", "#55A868"]
    for i, patch in enumerate(box["boxes"]):
        patch.set_facecolor(colors[i % n_methods])

    for sep in separator_positions[:-1]:
        ax.axvline(sep, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

    legend_elements = [
        Patch(facecolor="#DD8452", label=method2_label),
        Patch(facecolor="#4C72B0", label=method1_label),
    ]
    if n_methods == 3:
        legend_elements.append(Patch(facecolor="#55A868", label=method3_label))
    ax.legend(handles=legend_elements)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(labels=labels)

    ax.set_ylabel(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def creat_box_plot(result_path, file_name,
                   method1_col="abs_err_gnn", method2_col="abs_err_center",
                   method1_label="GNN", method2_label="Center of mass",
                   method3_col=None, method3_label=None):
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
                method1_col, method2_col, method1_label, method2_label,
                method3_col, method3_label)

    _plot_group(df, angle_par_dict, plot_angle_img_path, "MAE, °",
                method1_col, method2_col, method1_label, method2_label,
                method3_col, method3_label)

    _create_summary_table_plot(df, length_par_dict,
                               length_table_path, length_txt_path, "Length Measurements",
                               method1_col, method2_col, method1_label, method2_label,
                               method3_col, method3_label)

    _create_summary_table_plot(df, angle_par_dict,
                               angle_table_path, angle_txt_path, "Angle Measurements",
                               method1_col, method2_col, method1_label, method2_label,
                               method3_col, method3_label)

    aggregated_groups = {
        "Length measurements": list(length_par_dict.keys()),
        "Angle measurements": list(angle_par_dict.keys()),
    }
    _create_aggregated_norm_table(df, aggregated_groups,
                                   aggregated_table_path, aggregated_txt_path,
                                   method1_col, method2_col, method1_label, method2_label,
                                   method3_col, method3_label)

def _plot_robustness_heatmap(df, save_path):
    df = df.copy()
    df['gnn_vs_com_pct'] = (df['com_mean'] - df['gnn_mean']) / df['com_mean'] * 100

    pivot_gnn = df.pivot(index='sigma', columns='p', values='gnn_mean')
    pivot_pct = df.pivot(index='sigma', columns='p', values='gnn_vs_com_pct')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    specs = [
        (axes[0], pivot_gnn, "GNN MAE (mm)\nby noise level", ".3f", "YlOrRd"),
        (axes[1], pivot_pct, "GNN improvement over\nCenter of Mass (%)", ".1f", "YlGn"),
    ]
    for ax, pivot, title, fmt, cmap in specs:
        im = ax.imshow(pivot.values, cmap=cmap, aspect='auto')
        plt.colorbar(im, ax=ax)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(v).rstrip('0').rstrip('.') for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.1f}" for v in pivot.index])
        ax.set_xlabel("Perturbation probability (p)")
        ax.set_ylabel("Noise std (σ, mm)")
        ax.set_title(title)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:{fmt}}", ha='center', va='center', fontsize=8, color='black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def _plot_robustness_summary_table(df, save_path, save_txt_path):
    df = df.copy()
    df['gnn_vs_com_pct'] = (df['com_mean'] - df['gnn_mean']) / df['com_mean'] * 100

    agg = df.groupby('sigma').agg(
        gnn_mean=('gnn_mean', 'mean'),
        gnn_std=('gnn_mean', 'std'),
        com_mean=('com_mean', 'mean'),
        com_std=('com_mean', 'std'),
        imp_mean=('gnn_vs_com_pct', 'mean'),
        imp_std=('gnn_vs_com_pct', 'std'),
    ).reset_index()

    rows = []
    for _, r in agg.iterrows():
        rows.append([
            f"{r['sigma']:.1f}",
            f"{r['gnn_mean']:.3f} ± {r['gnn_std']:.3f}",
            f"{r['com_mean']:.3f} ± {r['com_std']:.3f}",
            f"{r['imp_mean']:.1f} ± {r['imp_std']:.1f}",
        ])

    col_labels = [
        "Noise std\n(σ, mm)",
        "GNN MAE\n(mean ± sd, mm)",
        "CoM MAE\n(mean ± sd, mm)",
        "GNN improvement\nover CoM (%)",
    ]

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(rows) + 1.0))
    ax.axis('off')

    table = ax.table(cellText=rows, colLabels=col_labels, loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(4)))

    for (row, col), cell in table.get_celld().items():
        cell.set_text_props(ha='center')
        if row > 0:
            cell.PAD = 0.06
    for col in range(4):
        table[(0, col)].set_height(table[(0, col)].get_height() * 1.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    with open(save_txt_path, 'w', encoding='utf-8') as f:
        headers = [
            "Noise std (sigma, mm)", "GNN MAE (mean +/- sd, mm)",
            "CoM MAE (mean +/- sd, mm)", "GNN improvement over CoM (%)",
        ]
        f.write('\t'.join(headers) + '\n')
        for row in rows:
            f.write('\t'.join(row) + '\n')


def plot_robustness_noise(robustness_folder, save_folder=None):
    if save_folder is None:
        save_folder = robustness_folder

    df = pd.read_csv(os.path.join(robustness_folder, "robustness_noise_sweep.csv"))

    _plot_robustness_heatmap(df, os.path.join(save_folder, "robustness_heatmap.png"))
    _plot_robustness_summary_table(
        df,
        os.path.join(save_folder, "robustness_summary_table.png"),
        os.path.join(save_folder, "robustness_summary_table.txt"),
    )


if __name__ == "__main__":

    result_path = r'C:\Users\Kamil\Aortic_valve\data\gnn_folder\results'
    # result_path = r'D:\science\Aortic_valve\GNN\results'
    file_name = "gnn_vs_center_comparison.csv"
    creat_box_plot(result_path, file_name)
