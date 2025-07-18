import pandas as pd
import numpy as np
from scipy.stats import kstest
import matplotlib.pyplot as plt

df = pd.read_excel('eval_results_r5colorcpdpp.xlsx', engine='openpyxl')
auto_data = df[df['ori'] == 'auto']['bcpdpp_jl'].dropna().astype(float).values
manual_data = df[df['ori'] == 'manual']['bcpdpp_jl'].dropna().astype(float).values
print("bcpdpp_jl (auto):", auto_data)
print("bcpdpp_jl (manual):", manual_data)
stat, p_value = kstest(auto_data, manual_data)
print("\nK-S test results:")
print(f"D statistic: {stat:.4f}")
print(f"p-value: {p_value:.4f}")
alpha = 0.05
if p_value < alpha:
    print(f"Reject null hypothesis (p < {alpha}): bcpdpp_jl distributions differ significantly between auto and manual.")
else:
    print(f"Cannot reject null hypothesis (p >= {alpha}): No sufficient evidence to prove bcpdpp_jl distributions differ.")
print(f"\nSupplementary statistics:")
print(f"auto mean: {np.mean(auto_data):.4f}, standard deviation: {np.std(auto_data):.4f}")
print(f"manual mean: {np.mean(manual_data):.4f}, standard deviation: {np.std(manual_data):.4f}")
def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)
    return x, y
x_auto, y_auto = ecdf(auto_data)
x_manual, y_manual = ecdf(manual_data)
plt.figure(figsize=(10, 6))
plt.plot(x_auto, y_auto, label='bcpdpp_jl (auto)', color='#1f77b4', linewidth=2)
plt.plot(x_manual, y_manual, label='bcpdpp_jl (manual)', color='#ff7f0e', linewidth=2)
plt.xlabel('Average Distance (bcpdpp_jl)')
plt.ylabel('Cumulative Probability')
plt.title('Empirical CDF of bcpdpp_jl: Auto vs Manual')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(auto_data, bins=10, alpha=0.5, label='bcpdpp_jl (auto)', color='#1f77b4', density=True)
plt.hist(manual_data, bins=10, alpha=0.5, label='bcpdpp_jl (manual)', color='#ff7f0e', density=True)
plt.xlabel('Average Distance (bcpdpp_jl)')
plt.ylabel('Density')
plt.title('Histogram of bcpdpp_jl: Auto vs Manual')
plt.legend()
plt.grid(True)
plt.show()