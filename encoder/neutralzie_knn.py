import torch
from torch.utils.data import random_split
from sklearn.cluster import KMeans
from scipy import stats as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
full_dataset = []

for i in range (3, 7):
    load_model = 'round'+str(i)+'.pt'
    # print(load_model)
    full_dataset = full_dataset+ torch.load(load_model).tolist()

kmeans = KMeans(n_clusters=10, random_state=42).fit(full_dataset)

res = kmeans.labels_
print(res)
print(res[0], res[12])

round3 = res[0:170]
round4 = res[170:340]
round5 = res[340:510]
round6 = res[510:680]

print('round3: ', round3, st.mode(res[0:170]))
print('round4: ', round4, st.mode(res[170:340]))
print('round5: ', round5, st.mode(res[340:510]))
print('round6: ', round6, st.mode(res[510:680]))

count = 3

for i in range(0, 170*4, 170):
    round = res[i: i+170]
    plt.figure()
    plt.xlim((0, 9))
    plt.ylim((0, 200))
    plt.title('Round ' + str(count))
    sns.histplot(data=round, bins=30)
    count+=1

plt.show()

# train_size = int(0.8 * len(full_dataset))
# test_size = len(full_dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

#
# for test_case in test_dataset:
#     dist = torch.norm(train_dataset - test_case, dim=1, p=None)
#     knn = dist.topk(3, largest=False)
#     print('kNN dist: {}, index: {}'.format(knn.values, knn.indices))