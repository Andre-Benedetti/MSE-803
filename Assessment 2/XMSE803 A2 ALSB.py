import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import os
import numpy as np

# 1. Setup
os.makedirs('Analysis_Outputs', exist_ok=True)
df = pd.read_csv('dataset_for_assignment_2.csv')
numerical_cols = ['Age', 'App Sessions', 'Distance Travelled (km)', 'Calories Burned']

print("\n--- Dataset Info (Fields and Data Types) ---")
print(df.info())

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Categorical Data Distribution ---")
categorical_cols = ['Gender', 'Location', 'Activity Level']
for col in categorical_cols:
    print(f"\nDistribution for {col}:")
    print(df[col].value_counts().to_string())

# 2. Interactive Outlier Review
print("--- Starting Manual Outlier Inspection ---")
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    if not outliers.empty:
        print(f"\nFound {len(outliers)} outliers in '{col}'")
        for index, row in outliers.iterrows():
            print(f"\n--- Row Index: {index} ---")
            print(row.to_string())
            
            try:
                decision = input("Action: [d]elete, [m]odify, [s]kip? ").lower()
                if decision == 'd':
                    df = df.drop(index)
                    print("Row deleted.")
                elif decision == 'm':
                    new_val = float(input(f"Enter new value for {col}: "))
                    df.at[index, col] = new_val
                    print(f"Row updated to {new_val}.")
                else:
                    print("Row skipped.")
            except EOFError:
                print("\nInput stream closed. Skipping remaining interactions.")
                break

# Save the manually cleaned dataset
df.to_csv('cleaned_dataset.csv', index=False)
print("\nCleaned dataset saved successfully.")

#3. Automated Analysis Pipeline
#Correlation Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.tight_layout()
plt.savefig('Analysis_Outputs/correlation_matrix.png')
plt.close()

# Regression Model
X = df[['App Sessions', 'Distance Travelled (km)']]
y = df['Calories Burned']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
predictions = model.predict(X_test)

# Calculate comprehensive metrics
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

print(f"\nRegression Performance Metrics:")
print(f"  R² Score: {r2:.4f}")
print(f"  MAE: {mae:.2f} calories")
print(f"  RMSE: {rmse:.2f} calories")
print(f"  MAPE: {mape:.1f}%")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.5, label='Actual Data')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label=f'Ideal Prediction (R²={r2:.2f})')
plt.xlabel('Actual Calories Burned')
plt.ylabel('Predicted Calories Burned')
plt.legend()
plt.savefig('Analysis_Outputs/regression_performance.png')
plt.close()

# Clustering
features = df[['App Sessions', 'Distance Travelled (km)', 'Calories Burned']]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(features)
df['Cluster'] = kmeans.labels_

plt.figure(figsize=(10, 6))
label_map = {0: 'Sedentary', 1: 'Moderate', 2: 'Active'}
for i in range(3):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['App Sessions'], cluster_data['Calories Burned'], 
                label=label_map[i], alpha=0.6)
plt.xlabel('App Sessions')
plt.ylabel('Calories Burned')
plt.legend()
plt.savefig('Analysis_Outputs/user_clusters.png')
plt.close()

# Perform Cross-Tabulation to validate K-Means clusters against original 'Activity Level'
df['Cluster_Label'] = df['Cluster'].map(label_map)
comparison = pd.crosstab(df['Cluster_Label'], df['Activity Level'], rownames=['K-Means'], colnames=['Original'])

print("\n--- Cluster vs Original Label Distribution ---")
print(comparison)

# Clustering Performance Metrics (MOVED HERE - AFTER Cluster_Label is created)
silhouette = silhouette_score(features, kmeans.labels_)
ari = adjusted_rand_score(df['Activity Level'], df['Cluster_Label'])

print(f"\nClustering Performance Metrics:")
print(f"  Silhouette Score: {silhouette:.4f}")
print(f"  Adjusted Rand Index: {ari:.4f}")

# ============================================
# CULTURAL BIAS DETECTION
# ============================================

print("\n=== CULTURAL BIAS DETECTION ===")

# Create age groups if not exists
if 'Age_Group' not in df.columns:
    df['Age_Group'] = pd.cut(df['Age'], 
                             bins=[18, 25, 35, 45, 60], 
                             labels=['18-25', '26-35', '36-45', '46-60'])

def detect_cultural_bias(df, feature_cols, target_col):
    """Detect potential cultural bias across demographic groups."""
    bias_results = []
    demographic_groups = ['Location', 'Gender', 'Age_Group']
    
    for group in demographic_groups:
        print(f"\nAnalyzing bias by {group}:")
        
        for category in df[group].unique():
            mask = df[group] == category
            group_data = df[mask]
            
            if len(group_data) < 30:
                continue
                
            X = group_data[feature_cols]
            y = group_data[target_col]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            model = LinearRegression().fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            r2 = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            
            bias_results.append({
                'Group': group,
                'Category': category,
                'R2_Score': r2,
                'MAE': mae,
                'Sample_Size': len(group_data)
            })
            
            print(f"  {category}: R²={r2:.3f}, MAE={mae:.2f}, n={len(group_data)}")
    
    return pd.DataFrame(bias_results)

# Run bias detection
bias_df = detect_cultural_bias(df, ['App Sessions', 'Distance Travelled (km)'], 'Calories Burned')
bias_df.to_csv('Analysis_Outputs/cultural_bias_analysis.csv', index=False)
print("\n✓ cultural_bias_analysis.csv saved")

print("\nAll analyses completed and saved to /Analysis_Outputs.")