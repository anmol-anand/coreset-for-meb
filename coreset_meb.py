import numpy as np
import math
import matplotlib.pyplot as plt

# Returns a set of unit vectors in 2D space. For each vector, its radial angle is returned.
def get_theta_net(theta):
    assert 0 <= theta < 2 * np.pi
    net_size = math.ceil(2 * math.pi / theta)
    theta_net = np.ndarray(net_size, dtype=float)
    for j in range(net_size):
        theta_net[j] = theta * j
    return theta_net

def get_noramlized_scalar_product(phi, x, y):
    return (math.cos(phi) * x + math.sin(phi) * y) / math.sqrt(x ** 2 + y ** 2)

# @arg in_P is in Euclidean coordinates.
def get_coreset_meb(in_P, eps):
    assert isinstance(in_P, np.ndarray)
    assert in_P.shape[1] == 2
    theta_net = get_theta_net(math.sqrt(eps))
    coreset_meb = set()
    for j in range(theta_net.shape[0]):
        max_norm_scalar_pdt = -1
        min_norm_scalar_pdt = 1
        arg_max = -1
        arg_min = -1
        for i in range(in_P.shape[0]):
            phi = theta_net[j]
            x = in_P[i][0]
            y = in_P[i][1]
            norm_scalar_pdt = get_noramlized_scalar_product(phi, x, y)
            if norm_scalar_pdt > max_norm_scalar_pdt:
                max_norm_scalar_pdt = norm_scalar_pdt
                arg_max = i
            if norm_scalar_pdt < min_norm_scalar_pdt:
                min_norm_scalar_pdt = norm_scalar_pdt
                arg_min = i
        coreset_meb.add(arg_max)
        coreset_meb.add(arg_min)
    return coreset_meb

num_in_P = 100000
in_P = np.random.rand(num_in_P, 2).astype(float)
eps = 1e-3
coreset_meb = get_coreset_meb(in_P, eps)

print("Input points:")
print(in_P)
print("\n\n")

print("Coreset for MEB cost function:")
print(coreset_meb)
print("\n\n")

plt.scatter(in_P[:, 0], in_P[:, 1], color='blue', marker='o', label='Original Set of Points')
plt.scatter(in_P[:, 0][list(coreset_meb)], in_P[:, 1][list(coreset_meb)], color='red', marker='x', label='Points in Coreset')

plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'original set size: {num_in_P} || theta net size: {math.ceil(2 * math.pi / math.sqrt(eps))} || coreset size: {len(coreset_meb)}')
plt.grid(True)
plt.show()
