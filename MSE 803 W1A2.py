import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder

# 1. Load the dataset
try:
    df = pd.read_csv('Housing.csv')
    print("Dataset successfully loaded!\n")
except FileNotFoundError:
    print("Error: The file 'Housing.csv' was not found.")
    exit()

# 2. Preprocessing: Convert Categorical variables to Numeric
le = LabelEncoder()
# Select object and string columns to avoid future pandas warnings
categorical_cols = df.select_dtypes(include=['object', 'string']).columns

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Define target variable (y) and features (X)
y = df['price']
X = df.drop('price', axis=1)

# Add a constant (intercept) to the feature matrix
X = sm.add_constant(X)

# 3. Data Inspection: Combined Frequency and Mean Price Table
cols_to_analyze = [
    'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 
    'basement', 'hotwaterheating', 'airconditioning', 'parking', 
    'prefarea', 'furnishingstatus'
]

statistical_summary = df.describe()

# 4. Build a summary list for frequency and average price per category
summary_data = []
for col in cols_to_analyze:
    if col in df.columns:
        # Aggregate count and mean price simultaneously
        group = df.groupby(col)['price'].agg(['count', 'mean']).sort_index()
        
        for val, row in group.iterrows():
            summary_data.append({
                'Variable': col,
                'Value_Category': val,
                'Frequency': int(row['count']),
                'Average_Price': round(row['mean'], 2)
            })

df_frequency_mean = pd.DataFrame(summary_data)

# 5. Price Distribution (Histogram)
plt.figure(figsize=(8, 5))
# Scaling price to Millions for better readability
sns.histplot(df['price'] / 1e6, kde=True, color='blue')
plt.title("Distribution of Housing Prices")
plt.xlabel("Price (Millions)")
plt.ylabel("Number of Houses")
plt.savefig('housing_price_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Fit the Multiple Linear Regression Model (OLS)
model = sm.OLS(y, X).fit()
print("\n--- Regression Results ---")
print(model.summary())

# 7. T-Values Impact Analysis Chart
plt.figure(figsize=(12, 6))

# Get absolute t-values excluding 'const' and sort them
t_values = model.tvalues.drop('const').abs().sort_values(ascending=False)
t_df = t_values.reset_index()
t_df.columns = ['Variable', 'T_Value']

# Create the bar plot
ax = sns.barplot(x='Variable', y='T_Value', data=t_df, hue='Variable', palette='viridis', legend=False)

# Formatting labels and title
plt.title('Impact Analysis', fontsize=16)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Absolute T-Value', fontsize=12)
plt.xticks(rotation=45)

# Add R-squared and Prob (F-statistic) as text box inside the figure
stats_info = f"R-squared: {model.rsquared:.3f}\nProb (F-statistic): {model.f_pvalue:.2e}"
plt.text(0.95, 0.90, stats_info, transform=ax.transAxes, fontsize=12,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Save the chart
plt.tight_layout()
plt.savefig('t_values_impact_analysis.png', dpi=300)
plt.close()

# 8. Save all results to a single CSV file
# Extract OLS coefficients and statistics to a DataFrame
df_ols_coeffs = pd.DataFrame({
    'coef': model.params,
    'std err': model.bse,
    't': model.tvalues,
    'P>|t|': model.pvalues
})

# Collect overall model performance metrics
metrics_data = {
    'Metric': ['R-squared', 'Adj. R-squared', 'F-statistic', 'Prob (F-statistic)'],
    'Value': [model.rsquared, model.rsquared_adj, model.fvalue, model.f_pvalue]
}
df_metrics = pd.DataFrame(metrics_data)

# Write all sections into the CSV file using headers to separate them
with open('statistical_summary.csv', 'w') as f:
    f.write("SECTION,MODEL_OVERALL_METRICS\n")
    df_metrics.to_csv(f, index=False)
    
    f.write("SECTION,BASIC_STATISTICAL_SUMMARY\n")
    statistical_summary.to_csv(f)
    
    f.write("SECTION,FREQUENCY_AND_MEAN_PRICE_PER_CATEGORY\n")
    df_frequency_mean.to_csv(f, index=False)
    
    f.write("SECTION,CORRELATION_MATRIX\n")
    df.corr(numeric_only=True).to_csv(f)
    
    f.write("SECTION,OLS_REGRESSION_COEFFICIENTS\n")
    df_ols_coeffs.to_csv(f)
    
    f.write("SECTION,IMPACT_ANALYSIS_T_VALUES\n")
    t_values.to_csv(f, header=['T-Value'])

print("\nAll data processing complete.")
