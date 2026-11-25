"""Summary tables of the interobserver steps: each cell pools the values of its metrics over the
cases of its column and shows mean ± std (population std, ddof=0) or median [Q1–Q3]."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NOT_AVAILABLE = 'N/A'


def _format_cell(values, stat):
    if stat == 'median':
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        return f'{median:.2f} [{q1:.2f}–{q3:.2f}]'
    return f'{values.mean():.2f} ± {values.std():.2f}'


def build_summary(per_case, rows, columns, value_col):
    """per_case: long table with columns case, source, metric and value_col.
    rows: (label, metric keys pooled in the row). columns: (label, source, cases, stat).

    A cell is N/A when any of its metrics has no value at all in that column, so a row whose
    structures are incomplete (e.g. 'all' without BR) is never reported as a full result."""
    grouped = {}
    for case, source, metric, value in per_case[['case', 'source', 'metric', value_col]].itertuples(
            index=False):
        if not pd.isna(value):
            grouped.setdefault((source, metric), {})[case] = value

    table = []
    for label, keys in rows:
        row = {'Parameter': label}
        for column_label, source, cases, stat in columns:
            per_key = [[grouped[(source, key)][case] for case in cases
                        if case in grouped.get((source, key), {})] for key in keys]
            if any(len(values) == 0 for values in per_key):
                row[column_label] = NOT_AVAILABLE
            else:
                row[column_label] = _format_cell(
                    np.array([v for values in per_key for v in values]), stat)
        table.append(row)
    return pd.DataFrame(table)


def plot_summary(summary, table_path):
    """The shared plot_table has equal centred columns, which clips this wide table and hides
    the row hierarchy, so the layout is set here. Rows without leading spaces are section rows,
    rows ending with ', all' are group rows."""
    labels = [label.replace(' (n=', '\n(n=') for label in summary.columns]
    longest_label = max(len(str(label)) for label in summary.iloc[:, 0])
    first_width = max(2.1, 0.11 * longest_label)
    # Each value column is as wide as its longest text (a header line or a cell), since a
    # median [Q1–Q3] cell or a long header does not fit the width of a mean ± std column.
    other_widths = []
    for label, column in zip(labels[1:], summary.columns[1:]):
        longest = max([len(line) for line in label.split('\n')]
                      + [len(str(value)) for value in summary[column]])
        other_widths.append(max(1.35, 0.1 * longest + 0.3))
    col_widths = [first_width] + other_widths
    width = sum(col_widths)
    header_rows = 2  # column labels are split into two lines
    height = 0.3 * (len(summary) + header_rows + 1)

    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis('off')
    table = ax.table(cellText=summary.values.tolist(), colLabels=labels, cellLoc='center',
                     colWidths=[w / width for w in col_widths],
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.4)
        if row == 0:
            cell.set_height(cell.get_height() * header_rows)
            cell.set_facecolor('#d9d9d9')
            cell.set_text_props(fontweight='bold')
            continue
        label = summary.iloc[row - 1, 0]
        if col == 0:
            cell.set_text_props(ha='left')
            cell.PAD = 0.03
        if not label.startswith(' '):
            cell.set_facecolor('#d9d9d9')
            cell.set_text_props(fontweight='bold')
        elif label.endswith(', all'):
            cell.set_facecolor('#f0f0f0')

    fig.savefig(table_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
