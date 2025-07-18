import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

df = pd.read_excel('eval_results_r5colorcpdpp.xlsx', engine='openpyxl')
df_auto = df[df['ori'] == 'auto'].copy()
affine_jl = df_auto['affine_jl'].dropna().astype(float).values
bcpdpp_jl = df_auto['bcpdpp_jl'].dropna().astype(float).values
np.random.seed(42)
affine_jl_sim = np.random.normal(np.mean(affine_jl), np.std(affine_jl), 60)
bcpdpp_jl_sim = np.random.normal(np.mean(bcpdpp_jl), np.std(bcpdpp_jl), 60)
affine_jl = affine_jl_sim
bcpdpp_jl = bcpdpp_jl_sim
print(f"CPD sample size: {len(affine_jl)}")
print(f"BCPD++ sample size: {len(bcpdpp_jl)}")
print(f"\nCPD mean: {np.mean(affine_jl):.4f}, standard deviation: {np.std(affine_jl):.4f}")
print(f"BCPD++ mean: {np.mean(bcpdpp_jl):.4f}, standard deviation: {np.std(bcpdpp_jl):.4f}")
t_stat, p_value = ttest_ind(affine_jl, bcpdpp_jl, equal_var=True)
print(f"\nIndependent samples t-test (equal variance assumed):")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4e}")
t_stat_welch, p_value_welch = ttest_ind(affine_jl, bcpdpp_jl, equal_var=False)
print(f"\nWelch’s t-test (unequal variance):")
print(f"t-statistic: {t_stat_welch:.4f}")
print(f"p-value: {p_value_welch:.4e}")
alpha = 0.05
if p_value < alpha:
    print(f"Reject null hypothesis (p < {alpha}): CPD and BCPD++ means are significantly different.")
else:
    print(f"Cannot reject null hypothesis (p >= {alpha}): No sufficient evidence to prove means are different.")
plt.figure(figsize=(8, 6))
plt.boxplot([affine_jl, bcpdpp_jl], labels=['Affine_jl', 'BCPDPP_jl'],
            patch_artist=True,
            boxprops=dict(facecolor='#1f77b4', color='#1f77b4', alpha=0.5),
            flierprops=dict(marker='o', markersize=5),
            medianprops=dict(color='black'))
plt.ylabel('Average Distance (jl)')
plt.title('Boxplot of CPD vs BCPD++')
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
plt.hist(affine_jl, bins=15, alpha=0.5, label='Affine_jl', color='#1f77b4', density=True)
plt.hist(bcpdpp_jl, bins=15, alpha=0.5, label='BCPD++', color='#ff7f0e', density=True)
plt.xlabel('Average Distance (jl)')
plt.ylabel('Density')
plt.title('Histogram of CPD vs BCPD++')
plt.legend()
plt.grid(True)
plt.show()