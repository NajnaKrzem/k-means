from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

kmeans_ari = adjusted_rand_score(labels, assignments)                          
kmeans_nmi = normalized_mutual_info_score(labels, assignments)                  

print(f"k-means ARI: {kmeans_ari:.2f}, NMI: {kmeans_nmi:.2f}")
