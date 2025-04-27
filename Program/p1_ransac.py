# Punkt 1: Na podstawie opisu algorytmu RANSAC dostępnego tu:
# https://en.wikipedia.org/wiki/Random_sample_consensus
# napisać funkcję w języku Python do dopasowania płaszczyzny do chmury punktów.

import numpy as np

def ransac_plane(points, threshold=0.01, iterations=1000):
    best_inliers = []
    best_plane = None

    for _ in range(iterations):
        idx = np.random.choice(len(points), 3, replace=False)
        sample = points[idx]
        p1, p2, p3 = sample

        normal = np.cross(p2 - p1, p3 - p1)
        if np.linalg.norm(normal) == 0:
            continue

        normal = normal / np.linalg.norm(normal)
        d = -np.dot(normal, p1)

        distances = np.abs(np.dot(points, normal) + d)

        inliers = np.where(distances < threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (normal, d)

    return best_plane, best_inliers
