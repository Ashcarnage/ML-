import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
data = pd.read_csv("kmeans_Q1.csv")
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(data['Points'],data['Assists'],data['Rebounds'],color='b')

ax.set_xlabel('Points')
ax.set_ylabel('Assists')
ax.set_zlabel('Rebounds')

plt.show()

class KMeansModel():
    def __init__(self,data,k_max):
        self.data = data
        self.k_max = k_max
        self.history = []
    def fit(self):
        for k in range(2,self.k_max+1):
            pass
    def initialize_centroid(self,k):
        return self.data[np.random.choice(self.data.shape[0],k,replace=False)] 
    def assign_labels(self, centroids):
        
    



model = KMeansModel(data)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# class KMeans:
#     def __init__(self, data, k_max, max_iters=100, tol=1e-4):
#         self.data = data
#         self.k_max = k_max
#         self.max_iters = max_iters
#         self.tol = tol
#         self.history = []

#     def fit(self):
#         for k in range(2, self.k_max + 1):
#             centroids = self.initialize_centroids(k)
#             for _ in range(self.max_iters):
#                 labels = self.assign_labels(centroids)
#                 new_centroids = self.update_centroids(labels, k)

#                 # Check for convergence
#                 if np.all(np.abs(new_centroids - centroids) < self.tol):
#                     break
#                 centroids = new_centroids
            
#             self.history.append((k, centroids, labels))

#     def initialize_centroids(self, k):
#         return self.data[np.random.choice(self.data.shape[0], k, replace=False)]

#     def assign_labels(self, centroids):
#         distances = np.linalg.norm(self.data[:, np.newaxis] - centroids, axis=2)
#         return np.argmin(distances, axis=1)

#     def update_centroids(self, labels, k):
#         return np.array([self.data[labels == i].mean(axis=0) for i in range(k)])

#     def plot_clusters(self):
#         for k, centroids, labels in self.history:
#             plt.figure()
#             plt.scatter(self.data[:, 0], self.data[:, 1], c=labels, cmap='viridis')
#             plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
#             plt.title(f'K-Means Clustering (k={k})')
#             plt.xlabel('Points')
#             plt.ylabel('Assists')
#             plt.show()

# # Sample data as a DataFrame
# data = {
#     "Points": [18.0, 19.0, 14.0, 14.0, 11.0, 20.0, 28.0, 30.0, 31.0, 35.0,
#                33.0, 25.0, 25.0, 27.0, 29.0, 30.0, 19.0, 23.0],
#     "Assists": [3.0, 4.0, 5.0, 4.0, 7.0, 8.0, 7.0, 6.0, 9.0, 12.0,
#                 14.0, 9.0, 4.0, 3.0, 3.0, 12.0, 15.0, 11.0],
#     "Rebounds": [15, 14, 10, 8, 14, 13, 9, 5, 4, 11,
#                  6, 5, 3, 8, 12, 7, 6, 5]
# }

# df = pd.DataFrame(data)
# data_array = df[['Points', 'Assists']].to_numpy()  # Use only Points and Assists for clustering

# # Run KMeans clustering
# kmeans = KMeans(data_array, k_max=10)
# kmeans.fit()
# kmeans.plot_clusters()
