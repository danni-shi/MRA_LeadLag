import numpy as np
import matplotlib.pyplot as plt

with open('../results/alignment.npy', 'rb') as f:
    error_list = np.load(f)
    acc_list = np.load(f)
    error_list_0 = np.load(f)
    acc_list_0 = np.load(f)

sigma_range = np.linspace(0.8,2.4,5) 
fig, ax = plt.subplots(figsize = (15,6))
ax.plot(sigma_range, error_list, label = 'with intermediate')
ax.plot(sigma_range, error_list_0, label = 'pairwise')
plt.grid()
plt.legend()
plt.title(f'Change of Alignment Error with Noise Level')
plt.savefig(f'../plots/align_error_0')

fig, ax = plt.subplots(figsize = (15,6))
ax.plot(sigma_range, acc_list, label = 'with intermediate')
ax.plot(sigma_range, acc_list_0, label = 'pairwise')
plt.grid()
plt.legend()
# plt.title(f'Change of Alignment Accuracy with Noise Level')
plt.savefig(f'../plots/align_acc_0')