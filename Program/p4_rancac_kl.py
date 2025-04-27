# Punkt 4: Dopasowanie płaszczyzny do każdej chmury (własne RANSAC) i analiza

import numpy as np
from p1_ransac import ransac_plane
from p2_load_xyz import load_xyz
from sklearn.cluster import KMeans

def is_horizontal(normal, tolerance=10):
    angle = np.degrees(np.arccos(np.abs(normal[2])))
    return angle < tolerance

def is_vertical(normal, tolerance=10):
    xy_norm = np.linalg.norm(normal[:2])
    angle = np.degrees(np.arccos(xy_norm))
    return angle < tolerance

points = load_xyz('dane_xyz/horizontal_plane.xyz')
##points = load_xyz('dane_xyz/cylinder.xyz')
##points = load_xyz('dane_xyz/vertical_plane.xyz.xyz')
2
kmeans = KMeans(n_clusters=3, random_state=0).fit(points)
labels = kmeans.labels_

for cluster_id in np.unique(labels):
    cluster_points = points[labels == cluster_id]

    print(f"\n--- Klastr {cluster_id} ---")

    plane, inliers = ransac_plane(cluster_points, threshold=0.01, iterations=500)

    if plane is None:
        print("Nie znaleziono płaszczyzny.")
        continue

    normal, d = plane
    print(f"Wejktor normalny: {normal}")

    distances = np.abs(np.dot(cluster_points, normal) + d)
    mean_distance = np.mean(distances)
    print(f"Średnia odległość: {mean_distance:.6f}")

    if mean_distance < 0.01:
        print("Klastr reprezentuje płaszczyznę.")
        if is_horizontal(normal):
            print("Płaszczyzna jest pozioma.")
        elif is_vertical(normal):
            print("Płaszczyzna jest pionowa.")
        else:
            print("Płaszczyzna ukośna.")
    else:
        print("To nie jest płaszczyzna.")
