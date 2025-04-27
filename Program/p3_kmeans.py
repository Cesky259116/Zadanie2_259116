# Punkt 3: Znaleźć rozłączne chmury punktów za pomocą algorytmu k-średnich (dla k=3).

import numpy as np
from sklearn.cluster import KMeans
from p2_load_xyz import load_xyz  # zakładam że masz plik punkt2

points = load_xyz('dane_xyz/cylinder.xyz')
##points = load_xyz('dane_xyz/horizontal_plane.xyz')
##points = load_xyz('dane_xyz/vertical_plane.xyz.xyz')

kmeans = KMeans(n_clusters=3, random_state=0).fit(points)
labels = kmeans.labels_

for cluster_id in np.unique(labels):
    cluster_points = points[labels == cluster_id]
    print(f"Klastr {cluster_id}: {len(cluster_points)} punktów")
