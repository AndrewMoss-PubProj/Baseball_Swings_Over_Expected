import pandas as pd
from collections import Counter
import numpy as np
from scipy.stats import t
import sklearn.metrics as metrics


def assignCountCluster(df):
    df['count_cluster'] = 0

    df.loc[(df['balls'] == 3) & (df['strikes'] == 0), 'count_cluster'] = 1

    df.loc[(df['balls'] == 0) & (df['strikes'] == 0), 'count_cluster'] = 2

    df.loc[(df['balls'] == 1) & (df['strikes'] == 0), 'count_cluster'] = 3
    df.loc[(df['balls'] == 2) & (df['strikes'] == 0), 'count_cluster'] = 3
    df.loc[(df['balls'] == 0) & (df['strikes'] == 1), 'count_cluster'] = 3

    df.loc[(df['balls'] == 0) & (df['strikes'] == 2), 'count_cluster'] = 4
    df.loc[(df['balls'] == 1) & (df['strikes'] == 1), 'count_cluster'] = 4

    df.loc[(df['balls'] == 3) & (df['strikes'] == 1), 'count_cluster'] = 5
    df.loc[(df['balls'] == 2) & (df['strikes'] == 1), 'count_cluster'] = 5
    df.loc[(df['balls'] == 1) & (df['strikes'] == 2), 'count_cluster'] = 5

    df.loc[(df['balls'] == 2) & (df['strikes'] == 2), 'count_cluster'] = 6

    df.loc[(df['balls'] == 3) & (df['strikes'] == 2), 'count_cluster'] = 7
    return df

def getNamesFromIDs(data):
    groupedData = data.groupby('pitcher').agg(pd.Series.mode)
    count = pd.Series(Counter(data['pitcher']))
    count = pd.DataFrame(count).rename(columns={0: 'Count'})
    count = count[count['Count'] > 300].sort_values(by='Count', ascending=False)
    count.reset_index(level=0, inplace=True)
    table = pd.DataFrame(groupedData.loc[:, 'player_name'])
    table.reset_index(level=0, inplace=True)
    table = pd.merge(count, table, how='right', left_on='index', right_on='pitcher')
    table = table[table['Count'].notna()]
    return table
