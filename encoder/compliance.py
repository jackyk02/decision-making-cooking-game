import torch
from torch.utils.data import random_split
from sklearn.cluster import KMeans
from scipy import stats as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def follow_algorithm_tip(res_id, round, df):
    # server cooks twice
    tmp = df[(df['ResponseId'] == res_id) & (df['round'] == round) & (df['worker'] == 2)]['action'].astype(np.int32)
    cnt = tmp.value_counts()[2] if 2 in tmp.value_counts() else 0
    return cnt == 2

def follow_human_tip(res_id, round, df):
    # server cooks once
    tmp = df[(df['ResponseId'] == res_id) & (df['round'] == round) & (df['worker'] == 2)]['action'].astype(np.int32)
    cnt = tmp.value_counts()[2] if 2 in tmp.value_counts() else 0
    return cnt == 1

def follow_baseline_tip(res_id, round, df):
    # sous-chef plates twice
    tmp = df[(df['ResponseId'] == res_id) & (df['round'] == round) & (df['worker'] == 1)]['action'].astype(np.int32)
    cnt = tmp.value_counts()[3] if 3 in tmp.value_counts() else 0
    return cnt == 2

def follow_unshown_tip(res_id, round, df):
    # server chops once
    tmp = df[(df['ResponseId'] == res_id) & (df['round'] == round) & (df['worker'] == 2)]['action'].astype(np.int32)
    cnt = tmp.value_counts()[1] if 1 in tmp.value_counts() else 0
    return cnt == 1



def getCompliance(round):
    df = pd.read_csv("data.csv")
    agg_df = df[df['round'] == round].groupby('ResponseId').agg(max)['treatment'].reset_index()
    tot = agg_df['treatment'].value_counts()
    print('round: ', round, tot)
    algo_tip = {'b1':0, 's1':0, 't1':0, 'control':0}
    human_tip = {'b1': 0, 's1': 0, 't1': 0, 'control': 0}
    base_tip = {'b1': 0, 's1': 0, 't1': 0, 'control': 0}
    unshown_tip = {'b1': 0, 's1': 0, 't1': 0, 'control': 0}
    for i in range(agg_df.shape[0]):
        res_id, treat = agg_df['ResponseId'][i], agg_df['treatment'][i]
        if follow_algorithm_tip(res_id, round, df):
            algo_tip[treat] += 1
        if follow_human_tip(res_id, round, df):
            human_tip[treat] += 1
        if follow_baseline_tip(res_id, round, df):
            base_tip[treat] += 1
        if follow_unshown_tip(res_id, round, df):
            unshown_tip[treat] += 1
    print(algo_tip)
    print(human_tip)
    print(base_tip)
    print(unshown_tip)


# getCompliance(6)
# df = pd.read_csv("data.csv")
# print(follow_algorithm_tip('R_09f8RCjNyCadwPv', 6, df))