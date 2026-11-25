"""Boxplots of the network errors for the paper figure: (a) geometry of the structures,
(b) morphometric measurements. One box per group of structures or per parameter; the values
of all its structures (cusps, commissures) over the network cases are pooled."""
import matplotlib.pyplot as plt
import numpy as np

BOX_COLOR = '#cfe2f3'
POINT_COLOR = 'black'
MEDIAN_COLOR = 'red'
JITTER = 0.08
SEED = 0

# Complex cases marked on both figures: case -> (colour, marker, legend label). Cohort 1 = n and p,
# Cohort 2 = g. Every value of these cases is
# marked, not only the outliers. Remove a line to drop a case, {} turns the marking off.
HIGHLIGHT_CASES = {
    'g61': ('#0072B2', 'D', 'Case A (Cohort 2)'),
    'p8': ('#009E73', '^', 'Case B (Cohort 1)'),
    'p60': ('#E69F00', 's', 'Case C (Cohort 1)'),
}


def _values(per_case, value_col, metrics, measure=None):
    """(values, cases) of the network for the pooled metrics."""
    subset = per_case[(per_case['source'] == 'DL') & per_case['metric'].isin(metrics)]
    if measure is not None:
        subset = subset[subset['measure'] == measure]
    subset = subset.dropna(subset=[value_col])
    return subset[value_col].to_numpy(dtype=float), subset['case'].to_numpy()


def _draw_boxes(ax, boxes, ylabel):
    """boxes: (label, (values, cases)). Median as a red line drawn over the points, every value as
    a jittered point (so the outlier markers of the box are not drawn); the highlighted cases
    have their own colour and marker."""
    boxes = [(label, pooled) for label, pooled in boxes if len(pooled[0])]
    data = [values for _, (values, _) in boxes]
    bplot = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.55,
                       medianprops=dict(color=MEDIAN_COLOR, linewidth=1.8, zorder=4))
    for patch in bplot['boxes']:
        patch.set_facecolor(BOX_COLOR)
    rng = np.random.default_rng(SEED)
    for i, (_, (values, cases)) in enumerate(boxes, start=1):
        x = rng.normal(i, JITTER, len(values))
        ordinary = ~np.isin(cases, list(HIGHLIGHT_CASES))
        ax.scatter(x[ordinary], values[ordinary], s=8, alpha=0.45,
                   color=POINT_COLOR, linewidths=0, zorder=3)
        for case, (color, marker, _) in HIGHLIGHT_CASES.items():
            own = cases == case
            ax.scatter(x[own], values[own], s=24, color=color, marker=marker,
                       edgecolors='black', linewidths=0.5, zorder=5)
    ax.set_xticks(range(1, len(boxes) + 1))
    ax.set_xticklabels([label for label, _ in boxes])
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)


def _legend(fig):
    handles = [plt.Line2D([], [], color=MEDIAN_COLOR, linewidth=1.8, label='Median'),
               plt.Line2D([], [], marker='o', linestyle='', color=POINT_COLOR, alpha=0.45,
                          markersize=4, label='Individual values')]
    handles += [plt.Line2D([], [], marker=marker, linestyle='', color=color, markersize=5.5,
                           markeredgecolor='black', markeredgewidth=0.5, label=label)
                for color, marker, label in HIGHLIGHT_CASES.values()]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False, fontsize=9,
               columnspacing=1.4, handletextpad=0.4)


def plot_geometry_boxplots(per_case, path, landmark_keys, curve_groups, dsc_key):
    """(a) per_case of the structures step. curve_groups: (label, curve keys)."""
    fig, (ax_dsc, ax_mm) = plt.subplots(1, 2, figsize=(7.5, 4.2),
                                        gridspec_kw={'width_ratios': [1, 4.5]})
    _draw_boxes(ax_dsc, [('Aortic root', _values(per_case, 'value', [dsc_key], 'DSC'))], 'DSC')
    boxes = [('Landmarks', _values(per_case, 'value', landmark_keys, 'PPD'))]
    boxes += [(label, _values(per_case, 'value', keys, 'CCD')) for label, keys in curve_groups]
    _draw_boxes(ax_mm, boxes, 'Absolute error, mm')
    _legend(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_morphometry_boxplots(per_case, path, length_groups, angle_groups):
    """(b) per_case of the morphometry step. *_groups: (label, metric keys)."""
    fig, (ax_mm, ax_deg) = plt.subplots(1, 2, figsize=(7.5, 4.2),
                                        gridspec_kw={'width_ratios': [len(length_groups),
                                                                      len(angle_groups)]})
    _draw_boxes(ax_mm, [(label, _values(per_case, 'abs_error', keys))
                        for label, keys in length_groups], 'Absolute error, mm')
    _draw_boxes(ax_deg, [(label, _values(per_case, 'abs_error', keys))
                         for label, keys in angle_groups], 'Absolute error, °')
    _legend(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=300)
    plt.close(fig)
