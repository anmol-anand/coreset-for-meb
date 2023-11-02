from coreset_meb import get_coreset_meb
from utils import diameter
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse

CORESET_INDEX = 0
LEVEL_INDEX = 1


parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', type=int, default=500, help='Number of samples in the original set')
parser.add_argument('--buffer_size', type=int, default=10, help='The maximum size of the buffer')
args = parser.parse_args()

num_samples = args.num_samples
buffer_size = args.buffer_size
epsilon_list = [i for i in range(1, 20)]
coreset_size_list = []
error_list = []

in_P = np.random.rand(num_samples, 2).astype(float)

for eps in epsilon_list:
    # each entry is a tuple of size two: (set (i.e. coreset), int ( i.e. level from bottom of the tree))
    coreset_meb_stack = []

    gamma = eps / math.log(num_samples, 2)

    for start_id in range(0, num_samples, buffer_size):
        batch_coreset_meb = get_coreset_meb(in_P[start_id : start_id + buffer_size], gamma)
        level_from_btm = 0
        while len(coreset_meb_stack) > 0 and coreset_meb_stack[-1][LEVEL_INDEX] == level_from_btm:
            batch_coreset_meb = np.concatenate((batch_coreset_meb, coreset_meb_stack[-1][CORESET_INDEX]), axis=0)
            coreset_meb_stack.pop()
            batch_coreset_meb = get_coreset_meb(batch_coreset_meb, gamma)
            level_from_btm = level_from_btm + 1
        assert(len(coreset_meb_stack) ==  0 or coreset_meb_stack[-1][LEVEL_INDEX] > level_from_btm)
        coreset_meb_stack.append((batch_coreset_meb, level_from_btm))

    assert(len(coreset_meb_stack) > 0)

    final_coreset_meb = coreset_meb_stack[-1][CORESET_INDEX]
    coreset_meb_stack.pop()
    while len(coreset_meb_stack) > 0:
        final_coreset_meb = np.concatenate((final_coreset_meb, coreset_meb_stack[-1][CORESET_INDEX]), axis=0)
        coreset_meb_stack.pop()
        final_coreset_meb = get_coreset_meb(final_coreset_meb, gamma)

    # approximate error: difference in radii
    error = (diameter(in_P) / 2 - diameter(final_coreset_meb) / 2) ** 2
    coreset_size_list.append(final_coreset_meb.shape[0])
    error_list.append(error)

    plt.scatter(in_P[:, 0], in_P[:, 1], color='blue', marker='o', label='Original Set of Points')
    plt.scatter(final_coreset_meb[:, 0], final_coreset_meb[:, 1], color='red', marker='x', label='Points in Coreset')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'original set size: {num_samples} || epsilon: {eps} || theta net size: {math.ceil(2 * math.pi / math.sqrt(eps))} || coreset size: {len(final_coreset_meb)}')
    plt.grid(True)
    plt.savefig(f'results_online/epsilon_{eps}.png')
    plt.clf()

plt.scatter(epsilon_list, error_list, color='green', marker='o', label='squared difference radius')
plt.plot(epsilon_list, error_list, color='green', linestyle='-', linewidth=1)
plt.xlabel('epsilon')
plt.ylabel('squared difference radius')
plt.grid(True)
plt.savefig('results_online/squared_difference_radius_v_epsilon.png')
plt.show()

plt.scatter(epsilon_list, coreset_size_list, color='purple', marker='o', label='coreset size')
plt.plot(epsilon_list, coreset_size_list, color='purple', linestyle='-', linewidth=1)
plt.xlabel('epsilon')
plt.ylabel('coreset size')
plt.grid(True)
plt.savefig('results_online/coreset_size_v_epsilon.png')
plt.show()