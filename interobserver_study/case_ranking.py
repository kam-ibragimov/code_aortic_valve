"""Per-case quality of the network against GT in one table, and a grading of the test cases from
the easiest to the hardest.

Each structure group gives one number per case (mean over its structures). Cases are ranked
within every group (1 = best); the case score is the mean rank over the ranked groups, and the
final rank orders cases by that score. The aortic root ranks (DSC, HD) are informative only and
do not enter the score."""
import pandas as pd

from data_preprocessing.text_worker import add_info_logging

from interobserver_study.sources import DL_GROUPS

ROOT_COLUMNS = {  # column: (metric, measure, higher is better)
    'AorticRoot_DSC': ('AorticRoot', 'DSC', True),
    'AorticRoot_HD': ('AorticRoot', 'HD', False),
}
RANKED_GROUPS = {  # column: (metrics averaged per case, measure); lower is better
    'Landmarks_PPD': (['R', 'L', 'N', 'RLC', 'RNC', 'LNC'], 'PPD'),
    'GH_CCD': (['RGH', 'LGH', 'NGH'], 'CCD'),
    'CI_CCD': (['RCI', 'LCI', 'NCI'], 'CCD'),
    'BR_CCD': (['BR'], 'CCD'),
}


def _case_group(case_name):
    for label, prefix in DL_GROUPS:
        if case_name.startswith(prefix):
            return label
    return 'other'


def build_case_ranking(per_case):
    """per_case: long table (case, source, metric, measure, value) of the structures step.
    Returns one row per network case, sorted from the easiest to the hardest case."""
    dl = per_case[per_case['source'] == 'DL']
    table = pd.DataFrame(index=sorted(dl['case'].unique()))
    table.index.name = 'case'
    table['group'] = [_case_group(case) for case in table.index]

    def values(metrics, measure):
        subset = dl[dl['metric'].isin(metrics) & (dl['measure'] == measure)]
        # A case lacking any structure of the group gets no group value.
        counts = subset.groupby('case')['value'].count()
        means = subset.groupby('case')['value'].mean()
        return means[counts == len(metrics)]

    for column, (metric, measure, _) in ROOT_COLUMNS.items():
        table[column] = values([metric], measure)
    for column, (metrics, measure) in RANKED_GROUPS.items():
        table[column] = values(metrics, measure)

    for column, (_, _, higher_better) in ROOT_COLUMNS.items():
        table[f'{column}_rank'] = table[column].rank(ascending=not higher_better, method='min')
    group_ranks = []
    for column in RANKED_GROUPS:
        table[f'{column}_rank'] = table[column].rank(ascending=True, method='min')
        group_ranks.append(f'{column}_rank')

    ranks = table[group_ranks]
    table['n_ranked_groups'] = ranks.notna().sum(axis=1)
    table['score_mean_rank'] = ranks.mean(axis=1)
    table['worst_group'] = ranks.idxmax(axis=1).str.replace('_rank', '', regex=False)
    table['final_rank'] = table['score_mean_rank'].rank(ascending=True, method='min')

    incomplete = table.index[table['n_ranked_groups'] < len(RANKED_GROUPS)].tolist()
    if incomplete:
        add_info_logging(f'Case ranking: groups missing in {incomplete}, score over the available '
                         f'groups only.', 'result_logger')

    table = table.sort_values(['final_rank', 'case']).reset_index()
    ordered = (['final_rank', 'case', 'group', 'score_mean_rank', 'worst_group']
               + list(RANKED_GROUPS) + group_ranks
               + list(ROOT_COLUMNS) + [f'{c}_rank' for c in ROOT_COLUMNS] + ['n_ranked_groups'])
    for column in ['final_rank'] + group_ranks + [f'{c}_rank' for c in ROOT_COLUMNS]:
        table[column] = table[column].astype('Int64')
    table['score_mean_rank'] = table['score_mean_rank'].round(2)
    return table[ordered]
