def update_centroids(data, num_clusters, assignments):
    centroids = []
    for c in range(num_clusters):
        cluster_points = data[np.array(assignments) == c] 
        if len(cluster_points) > 0:
            centroids.append(np.mean(cluster_points, axis=0))  
        else:
            centroids.append(np.random.random(data.shape[1]))  
    return np.array(centroids)
