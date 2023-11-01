import numpy as np

def diameter(points):
    max_distance_squared = 0

    for i in range(points.shape[0]):
        for j in range(i + 1, points.shape[0]):
            distance_squared = np.sum((points[i] - points[j])**2)
            max_distance_squared = max(max_distance_squared, distance_squared)

    return max_distance_squared
