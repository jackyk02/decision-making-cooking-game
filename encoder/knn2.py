import torch
from torch.utils.data import random_split
from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

full_dataset = np.array(torch.load('round3and4.pt').tolist())

kmeans = KMeans(n_clusters=6, random_state=0).fit(full_dataset)

res = kmeans.labels_
count = 3
print(res)

# for i in range(0, 170*4, 170):
#     round = res[i: i+170]
#     plt.figure()
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