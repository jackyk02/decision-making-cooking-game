import torch
from torch.utils.data import random_split
from sklearn.cluster import KMeans
import numpy as np

full_dataset = np.array(torch.load('tensor.pt').tolist())
kmeans = KMeans(n_clusters=3, random_state=0).fit(full_dataset)
print(kmeans.labels_)

# train_size = int(0.8 * len(full_dataset))
# test_size = len(full_dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

#
# for test_case in test_dataset:
#     dist = torch.norm(train_dataset - test_case, dim=1, p=None)
#     knn = dist.topk(3, largest=False)
#     print('kNN dist: {}, index: {}'.format(knn.values, knn.indices))