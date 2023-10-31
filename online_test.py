from coreset_meb import get_coreset_meb
import numpy as np
import math
import matplotlib.pyplot as plt

CORESET_INDEX = 0
LEVEL_INDEX = 1


# each entry is a tuple of size two: (set (/coreset), int (/level from bottom of the tree))
coreset_meb_stack = []

num_in_P = 100
eps = 1e-3
buffer_size = 5

in_P = np.random.rand(num_in_P, 2).astype(float)

gamma = eps / math.log(num_in_P, 2)

for start_id in range(0, num_in_P, buffer_size):
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

print("Input points:")
print(in_P)
print("\n\n")

print("Coreset for MEB cost function:")
print(final_coreset_meb)
print("\n\n")

plt.scatter(in_P[:, 0], in_P[:, 1], color='blue', marker='o', label='Original Set of Points')
plt.scatter(final_coreset_meb[:, 0], final_coreset_meb[:, 1], color='red', marker='x', label='Points in Coreset')

plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'original set size: {num_in_P} || theta net size: {math.ceil(2 * math.pi / math.sqrt(eps))} || coreset size: {len(final_coreset_meb)}')
plt.grid(True)
plt.show()
