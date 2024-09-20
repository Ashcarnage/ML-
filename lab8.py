import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
Data = pd.read_csv("fruits.csv")
X0 = Data.iloc[:,1:2]
X1 = Data.iloc[:,2:3]
data = np.hstack([X0,X1])
Y = Data.iloc[:,-1]
Y = np.where(Y=="Apple",1,0)

class KNN:
    def __init__(self, k):
        self.k = k
    def fit(self,X,Y):
        self.X_train = X
        self.y_train = Y
    def predict(self, X):
        return np.array([self._predict(x) for x in X])
    def _predict(self,x):
        # dist = self.euclidian_distance(self.X_train,x)
        dist = self.manhattan_distance(self.X_train,x)
        k_nearest = self.get_k_nearest(dist)
        return self.majority_vote(k_nearest)
    
    def euclidian_distance(self,X,x):
        return np.sqrt(np.sum((X-x)**2, axis = 1))
    def manhattan_distance(self,X,x):
        return np.sum(np.abs(X-x), axis=1)
    
    def get_k_nearest(self,dist):
        k_nearest = []
        print(dist)
        for i in range(self.k):
            min_idx = np.argmin(dist)
            k_nearest.append(self.y_train[min_idx])
            dist[min_idx] = 100000
        return k_nearest
    def majority_vote(self,k_nearest):
        label_count = {}
        for label in k_nearest:
            if label in label_count:
                label_count[label]+=1
            else:
                label_count[label] = 1
        return max(label_count,key = label_count.get)

knn = KNN(3)
knn.fit(data,Y)
pred = knn.predict(data)

# Plot results
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(data[:, 0], Y, data[:, 1], c=pred, cmap='coolwarm', s=50)
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
plt.colorbar(scatter, label='Cluster')
plt.title('3D KNN Clustering Results')
plt.show()