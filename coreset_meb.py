import numpy as np
import math

# Returns a set of unit vectors in 2D space. For each vector, its radial angle is returned.
def get_theta_net(theta):
    assert 0 <= theta < 2 * np.pi
    net_size = math.ceil(2 * math.pi / theta)
    theta_net = np.ndarray(net_size, dtype=float)
    for j in range(net_size):
        theta_net[j] = theta * j
    return theta_net

def get_scalar_product(phi, x, y):
    return math.cos(phi) * x + math.sin(phi) * y

# @arg in_P is in Euclidean coordinates.
def get_coreset_meb(in_P, eps):
    assert isinstance(in_P, np.ndarray)
    assert in_P.shape[1] == 2
    theta_net = get_theta_net(math.sqrt(eps))
    coreset_meb = set()
    for j in range(theta_net.shape[0]):
        max_scalar_pdt = -1
        min_scalar_pdt = 1
        arg_max = -1
        arg_min = -1
        for i in range(in_P.shape[0]):
            phi = theta_net[j]
            x = in_P[i][0]
            y = in_P[i][1]
            scalar_pdt = get_scalar_product(phi, x, y)
            if scalar_pdt > max_scalar_pdt:
                max_scalar_pdt = scalar_pdt
                arg_max = i
            if scalar_pdt < min_scalar_pdt:
                min_scalar_pdt = scalar_pdt
                arg_min = i
        coreset_meb.add(arg_max)
        coreset_meb.add(arg_min)
    return in_P[list(coreset_meb)]
