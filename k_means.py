import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

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
