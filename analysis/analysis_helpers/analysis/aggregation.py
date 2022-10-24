import sklearn.metrics.pairwise
import pandas
import numpy as np


def get_dists_ranks(df_input, range_criteria, domain_criteria):
    df_test = df_input[(df_input['game_set'] == 'test') & df_input['player_category'].isin(range_criteria)].sort_values(['player_category', 'player_name'])
    df_val = df_input[(df_input['game_set'] == 'validate') & df_input['player_category'].isin(domain_criteria)].sort_values(['player_category', 'player_name'])

    df_test = df_test[df_test['player_name'].isin(df_val['player_name'])].sort_values(['player_category', 'player_name'])

    df_val_in = df_val[df_val['player_name'].isin(df_test['player_name'])].sort_values(['player_category', 'player_name'])
    df_val_out = df_val[~df_val['player_name'].isin(df_test['player_name'])]

    if np.equal(df_test['player_name'].values, df_val_in['player_name'].values).mean() < 1.0:
        raise RuntimeError

    if len(df_val_out) > 0:
        stacked_domain = np.concatenate([np.stack(df_val_in['player_vec']), np.stack(df_val_out['player_vec'])], axis = 0)
    else:
        stacked_domain =  np.stack(df_val_in['player_vec'])

    dists = sklearn.metrics.pairwise.cosine_distances(
        np.stack(df_test['player_vec']),
        stacked_domain,
    )
    return np.abs(dists.argmin(axis = 1) - np.array(range(dists.shape[0]))), stacked_domain

def get_dists_matrix(df_input):
    df_test = df_input[df_input['game_set'] == 'test'].sort_values(['player_category', 'player_name'])
    df_val = df_input[df_input['game_set'] == 'validate'].sort_values(['player_category', 'player_name'])

    df_val = df_val[df_val['player_name'].isin(df_test['player_name'])]
    df_test = df_test[df_test['player_name'].isin(df_val['player_name'])]

    df_val = df_val.drop_duplicates('player_name')
    df_test = df_test.drop_duplicates('player_name')

    if np.equal(df_test['player_name'].values, df_val['player_name'].values).mean() < 1.0:
        raise RuntimeError

    dists = sklearn.metrics.pairwise.cosine_distances(
        np.stack(df_test['player_vec']),
        np.stack(df_val['player_vec']),
    )
    return pandas.DataFrame(dists, index = df_test['player_name'], columns=df_test['player_name'])
    #return df_compare.values.argmin(axis = 1) == np.array(range(len(df_val['player_name'].unique())))
