import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# 1. Load the dataset
try:
    df = pd.read_csv('age_networth.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: The file 'age_networth.csv' was not found.")
    exit()

# 2. Select columns
col_x = 'Age'
col_y = 'Net Worth'
df_clean = df[[col_x, col_y]].dropna()

# --- 3. OUTLIER DETECTION (IQR Method) ---
# Identify outliers specifically for Net Worth
Q1 = df_clean[col_y].quantile(0.25)
Q3 = df_clean[col_y].quantile(0.75)
IQR = Q3 - Q1

# Define bounds (1.5 * IQR is the standard multiplier)
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Separate outliers and normal data
outliers = df_clean[(df_clean[col_y] < lower_bound) | (df_clean[col_y] > upper_bound)]
df_normal = df_clean[(df_clean[col_y] >= lower_bound) & (df_clean[col_y] <= upper_bound)]

print(f"\n--- Outlier Detection Results ---")
print(f"Lower Bound: {lower_bound:.2f} | Upper Bound: {upper_bound:.2f}")
print(f"Total Outliers detected: {len(outliers)}")

# 4. Correlation on cleaned data (optional: use df_normal for a "purer" correlation)
r_coef, p_value = pearsonr(df_clean[col_x], df_clean[col_y])

# 5. Visualization
plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")

# Plot normal data
sns.scatterplot(x=col_x, y=col_y, data=df_normal, color='teal', alpha=0.5, label='Normal Data')

# Highlight Outliers in Red
sns.scatterplot(x=col_x, y=col_y, data=outliers, color='red', marker='X', s=100, label='Outliers')

# Trend line (based on all data)
sns.regplot(x=col_x, y=col_y, data=df_clean, scatter=False, color='darkred', label=f'Overall Trend (r={r_coef:.2f})')

# 6. Formatting
plt.title('Age vs Net Worth: Identifying Outliers', fontsize=15)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Net Worth', fontsize=12)
plt.legend()



# 7. Print the specific outlier values
if not outliers.empty:
    print("\nTop 5 Outlier Records:")
    print(outliers.sort_values(by=col_y, ascending=False).head())

plt.tight_layout()
plt.savefig('outlier_analysis.png', dpi=300)
plt.show()