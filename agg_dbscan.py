from sklearn.cluster import AgglomerativeClustering, DBSCAN

# Klasteryzacja hierarchiczna
agg = AgglomerativeClustering(n_clusters=num_clusters)
agg_assignments = agg.fit_predict(data)
agg_ari = adjusted_rand_score(labels, agg_assignments)
agg_nmi = normalized_mutual_info_score(labels, agg_assignments)
print(f"Agglomerative Clustering ARI: {agg_ari:.2f}, NMI: {agg_nmi:.2f}")

# Klasteryzacja DBSCAN
dbscan = DBSCAN(eps=15, min_samples=5)
dbscan_assignments = dbscan.fit_predict(data)

dbscan_assignments[dbscan_assignments == -1] = num_clusters  # Oznaczamy szum jako nowy klaster

dbscan_ari = adjusted_rand_score(labels, dbscan_assignments)
dbscan_nmi = normalized_mutual_info_score(labels, dbscan_assignments)
print(f"DBSCAN ARI: {dbscan_ari:.2f}, NMI: {dbscan_nmi:.2f}")
