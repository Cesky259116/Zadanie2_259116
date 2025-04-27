import numpy as np
from sklearn.cluster import DBSCAN
from p2_load_xyz import load_xyz
from pyransac3d import Plane

def is_horizontal(normal, tolerance=10):
    angle = np.degrees(np.arccos(np.abs(normal[2])))
    return angle < tolerance

def is_vertical(normal, tolerance=10):
    xy_norm = np.linalg.norm(normal[:2])
    angle = np.degrees(np.arccos(xy_norm))
    return angle < tolerance

# points = load_xyz('dane_xyz/cylinder.xyz')
points = load_xyz('dane_xyz/horizontal_plane.xyz')
# points = load_xyz('dane_xyz/vertical_plane.xyz.xyz')

# Klastrowanie za pomocą DBSCAN
dbscan = DBSCAN(eps=0.15, min_samples=40).fit(points)  # Zwiększenie eps i min_samples
labels = dbscan.labels_

plane_model = Plane()

for cluster_id in np.unique(labels):
    if cluster_id == -1:
        continue

    cluster_points = points[labels == cluster_id]

    print(f"\n--- Klastr {cluster_id} ---")

    try:
        # Dopasowanie płaszczyzny do punktów w klastrze
        plane_params, inlier_indices = plane_model.fit(cluster_points, thresh=0.01)

        # Sprawdzenie, ile wartości zostało zwróconych
        if len(plane_params) == 4:
            a, b, c, d = plane_params
            normal = np.array([a, b, c])

            print(f"Wejktor normalny: {normal}")

            distances = np.abs(np.dot(cluster_points, normal) + d)
            mean_distance = np.mean(distances)
            print(f"Średnia odległość: {mean_distance:.6f}")

            if mean_distance < 0.01:
                print("Płaszczyzna wykryta.")
                if is_horizontal(normal):
                    print("Płaszczyzna jest pozioma.")
                elif is_vertical(normal):
                    print("Płaszczyzna jest pionowa.")
                else:
                    print("Płaszczyzna ukośna.")
            else:
                print("Brak płaszczyzny.")
        else:
            print(f"Zwrócone wartości: {plane_params}")
            print("Problem z dopasowaniem płaszczyzny. Zwrócono niewłaściwą liczbę wartości.")

    except Exception as e:
        print(f"Problem z dopasowaniem płaszczyzny: {e}")