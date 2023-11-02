from coreset_meb import get_coreset_meb
from utils import diameter
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', type=int, default=500, help='Number of samples in the original set')
args = parser.parse_args()

num_samples = args.num_samples
epsilon_list = [0.5 * i for i in range(1, 21)]
coreset_size_list = []
error_list = []

in_P = np.random.rand(num_samples, 2).astype(float)

for eps in epsilon_list:
    coreset_meb = get_coreset_meb(in_P, eps)

    # approximate error: difference in radii
    error = (diameter(in_P) / 2 - diameter(coreset_meb) / 2) ** 2
    coreset_size_list.append(coreset_meb.shape[0])
    error_list.append(error)

    plt.scatter(in_P[:, 0], in_P[:, 1], color='blue', marker='o', label='Original Set of Points')
    plt.scatter(coreset_meb[:, 0], coreset_meb[:, 1], color='red', marker='x', label='Points in Coreset')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'original set size: {num_samples} || epsilon: {eps} || theta net size: {math.ceil(2 * math.pi / math.sqrt(eps))} || coreset size: {len(coreset_meb)}')
    plt.grid(True)
    plt.savefig(f'results_offline/epsilon_{eps}.png')
    plt.clf()

plt.scatter(epsilon_list, error_list, color='green', marker='o', label='squared difference radius')
plt.plot(epsilon_list, error_list, color='green', linestyle='-', linewidth=1)
plt.xlabel('epsilon')
plt.ylabel('squared difference radius')
plt.grid(True)
plt.savefig('results_offline/squared_difference_radius_v_epsilon.png')
plt.show()

plt.scatter(epsilon_list, coreset_size_list, color='purple', marker='o', label='coreset size')
plt.plot(epsilon_list, coreset_size_list, color='purple', linestyle='-', linewidth=1)
plt.xlabel('epsilon')
plt.ylabel('coreset size')
plt.grid(True)
plt.savefig('results_offline/coreset_size_v_epsilon.png')
plt.show()