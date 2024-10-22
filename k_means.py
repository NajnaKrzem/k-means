import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

def update_assignments(data, centroids):
    assignments = []
    for point in data:
        distances = np.linalg.norm(point - centroids, axis=1) 
        assignments.append(np.argmin(distances))  
    return assignments

def update_centroids(data, num_clusters, assignments):
    centroids = []
    for c in range(num_clusters):
        cluster_points = data[np.array(assignments) == c] 
        if len(cluster_points) > 0:
            centroids.append(np.mean(cluster_points, axis=0))  
        else:
            centroids.append(np.random.random(data.shape[1]))  
    return np.array(centroids)

data, labels = load_digits(return_X_y=True)
(n_samples, n_features), n_digits = data.shape, np.unique(labels).size
print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")

num_clusters = n_digits

centroids = data[np.random.choice(len(data), num_clusters, replace=False)]

iteration = 0
while True:
    iteration += 1
    previous_centroids = centroids.copy() 
    assignments = update_assignments(data, centroids)  
    centroids = update_centroids(data, num_clusters, assignments) 

    if np.array_equal(centroids, previous_centroids):
        print(f"Centroidy przestały się zmieniać po {iteration} iteracjach.")
        break

pca = PCA(2)
data_2d = pca.fit_transform(data)
centroids_2d = pca.transform(centroids)

plt.scatter(data_2d[:, 0], data_2d[:, 1], c=assignments, cmap='viridis', s=5)
plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], color='red', marker='*', s=100)
plt.show()
