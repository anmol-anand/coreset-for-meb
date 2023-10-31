from coreset_meb import get_coreset_meb
import numpy as np
import math
import matplotlib.pyplot as plt

num_in_P = 100
eps = 1e-3

in_P = np.random.rand(num_in_P, 2).astype(float)

coreset_meb = get_coreset_meb(in_P, eps)

print("Input points:")
print(in_P)
print("\n\n")

print("Coreset for MEB cost function:")
print(coreset_meb)
print("\n\n")

plt.scatter(in_P[:, 0], in_P[:, 1], color='blue', marker='o', label='Original Set of Points')
plt.scatter(coreset_meb[:, 0], coreset_meb[:, 1], color='red', marker='x', label='Points in Coreset')

plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'original set size: {num_in_P} || theta net size: {math.ceil(2 * math.pi / math.sqrt(eps))} || coreset size: {len(coreset_meb)}')
plt.grid(True)
plt.show()
