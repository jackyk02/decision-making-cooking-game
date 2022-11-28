import torch
from torch.utils.data import random_split
from sklearn.cluster import KMeans
from scipy import stats as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
full_dataset = []

for i in range (6, 7):
    load_model = 'round'+str(i)+'.pt'
    # print(load_model)
    full_dataset = full_dataset+ torch.load(load_model).tolist()

num_cluster = 4
kmeans = KMeans(n_clusters=num_cluster, random_state=42).fit(full_dataset)

res = kmeans.labels_
print(res)


def follow_algorithm_tip(res_id, df):
    # server cooks twice
    tmp = df[(df['ResponseId'] == res_id) & (df['worker'] == 2)]['action'].astype(np.int32)
    cnt = tmp.value_counts()[2] if 2 in tmp.value_counts() else 0
    return cnt == 2

def follow_human_tip(res_id, df):
    # server cooks once
    tmp = df[(df['ResponseId'] == res_id) & (df['worker'] == 2)]['action'].astype(np.int32)
    cnt = tmp.value_counts()[2] if 2 in tmp.value_counts() else 0
    return cnt == 1

def follow_baseline_tip(res_id, df):
    # sous-chef plates twice
    tmp = df[(df['ResponseId'] == res_id)& (df['worker'] == 1)]['action'].astype(np.int32)
    cnt = tmp.value_counts()[3] if 3 in tmp.value_counts() else 0
    return cnt == 2

def follow_unshown_tip(res_id, df):
    # server chops once
    tmp = df[(df['ResponseId'] == res_id) & (df['worker'] == 2)]['action'].astype(np.int32)
    cnt = tmp.value_counts()[1] if 1 in tmp.value_counts() else 0
    return cnt == 1

def follow_round12_algorithm_tip(res_id, df):
    #Chef shouldn't plate
    tmp = df[(df['ResponseId'] == res_id) & (df['worker'] == 0)]['action'].astype(np.int32)
    cnt = tmp.value_counts()[3] if 3 in tmp.value_counts() else 0
    return cnt == 0

def follow_round12_baseline_tip(res_id, df):
    #Chef chops once
    tmp = df[(df['ResponseId'] == res_id) & (df['worker'] == 0)]['action'].astype(np.int32)
    cnt = tmp.value_counts()[1] if 1 in tmp.value_counts() else 0
    return cnt == 1

# def follow_round12_human_tip(res_id, df):
#     #Leave workers idle


def analyzeTipCompliance():
    round = 6
    tip_name = ['algorithm_tip', 'human_tip', 'baseline_tip', 'unshown_tip']
    df = pd.read_csv('data.csv')
    df = df[df['round'] == round]
    id_list = df.groupby('ResponseId').agg(max).reset_index()
    label_cnt = np.bincount(res)
    tip_compliance_distribution = []
    print(label_cnt)
    for k in range(num_cluster):
        tip_count = [0]*4
        lst = np.where(res == k)[0]
        for idx in lst:
            res_id = id_list['ResponseId'][idx]
            if follow_algorithm_tip(res_id, df):
                tip_count[0] += 1
            if follow_human_tip(res_id, df):
                tip_count[1] += 1
            if follow_baseline_tip(res_id, df):
                tip_count[2] += 1
            if follow_unshown_tip(res_id, df):
                tip_count[3] += 1
        tip_compliance_distribution.append(tip_count)
        print('========cluster label:' + str(k) + '=============')
        for i in range(4):
            print(tip_name[i], ': ', tip_count[i]/label_cnt[k], end=' ' if i != 3 else '\n')
        max_pct = int(max(tip_count) / label_cnt[k] * 10000)/100
        max_tip = []
        for i in range(4):
            if tip_count[i] == max(tip_count):
                max_tip.append(tip_name[i])
        print('max tip: ', max_tip, ' with compliance rate ' + str(max_pct) + '%')




analyzeTipCompliance()


def analyze():
    df = pd.read_csv("trace.csv")
    round_list = [1, 2]
    start_round = 1
    num_items_each_round = 170
    workers = ['Chef', 'Sous Chef', 'Server']
    res_df = df[df['round'].isin(round_list)].groupby('ResponseId').agg(max).reset_index()
    df = df[np.invert(df['simplified_action'].isnull())]
    for label in range(num_cluster):
        print('=======analysis for cluster ', label, '===============')
        idx = np.where(res==label)[0][0]
        round = (idx // num_items_each_round) + start_round
        idx = idx % num_items_each_round
        res_id = res_df.iloc[idx]['ResponseId']
        print('ResponseID:', res_id, 'round:', round)
        for worker in workers:
            print(worker, 'actions: ')
            worker_id = workers.index(worker)
            action = df[(df['ResponseId'] == res_id) & (df['round'] == round) & (df['worker'] == worker_id)]['simplified_action'].astype(np.int32)
            action = action.to_numpy()
            action = action[action != 0]
            print(action)


#analyze()




# round3 = res[0:170]
# round4 = res[170:340]
# round5 = res[340:510]
# round6 = res[510:680]
#

# print('round1: ', round1, st.mode(res[0:170]))
# print('round2: ', round2, st.mode(res[170:340]))
# print('round3: ', round3, st.mode(res[0:170]))
# print('round4: ', round4, st.mode(res[170:340]))
# print('round5: ', round5, st.mode(res[340:510]))
# print('round6: ', round6, st.mode(res[510:680]))
#
# count = 3
#
# for i in range(0, 170*4, 170):
#     round = res[i: i+170]
#     plt.figure()
#     plt.xlim((0, 9))
#     plt.ylim((0, 200))
#     plt.title('Round ' + str(count))
#     sns.histplot(data=round, bins=30)
#     count+=1
#
# plt.show()

# train_size = int(0.8 * len(full_dataset))
# test_size = len(full_dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

#
# for test_case in test_dataset:
#     dist = torch.norm(train_dataset - test_case, dim=1, p=None)
#     knn = dist.topk(3, largest=False)
#     print('kNN dist: {}, index: {}'.format(knn.values, knn.indices))