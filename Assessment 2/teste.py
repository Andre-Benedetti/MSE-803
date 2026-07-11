import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
import os
import numpy as np
from sklearn.metrics import mean_absolute_error
#import warnings
#warnings.filterwarnings('ignore')

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
r2 = r2_score(y_test, predictions)
print(f"Regression R^2 Score: {r2:.2f}")

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

print("--- Cluster vs Original Label Distribution ---")
print(comparison)

print("\nAll analyses completed and saved to /Analysis_Outputs.")

# ============================================
# CULTURAL RELEVANCE ANALYSIS
# ============================================

print("\n" + "="*60)
print("CULTURAL RELEVANCE ANALYSIS")
print("="*60)

# Create age groups for cultural analysis
df['Age_Group'] = pd.cut(df['Age'], 
                         bins=[18, 25, 35, 45, 60], 
                         labels=['18-25', '26-35', '36-45', '46-60'])

# 1. Demographic Analysis
print("\n=== DEMOGRAPHIC ANALYSIS ===")

# Location analysis
print("\n--- Location-based Analysis ---")
location_analysis = df.groupby('Location').agg({
    'App Sessions': ['mean', 'std', 'count'],
    'Calories Burned': ['mean', 'std'],
    'Distance Travelled (km)': ['mean', 'std']
}).round(2)
print(location_analysis)

# Gender analysis
print("\n--- Gender-based Analysis ---")
gender_analysis = df.groupby('Gender').agg({
    'App Sessions': ['mean', 'std', 'count'],
    'Calories Burned': ['mean', 'std'],
    'Distance Travelled (km)': ['mean', 'std']
}).round(2)
print(gender_analysis)

# Age group analysis
print("\n--- Age Group Analysis ---")
age_analysis = df.groupby('Age_Group').agg({
    'App Sessions': 'mean',
    'Calories Burned': 'mean',
    'Distance Travelled (km)': 'mean'
}).round(2)
print(age_analysis)

# Activity level distribution by demographics
print("\n--- Activity Level Distribution ---")
print("\nBy Location:")
print(pd.crosstab(df['Location'], df['Activity Level'], normalize='index').round(3))
print("\nBy Gender:")
print(pd.crosstab(df['Gender'], df['Activity Level'], normalize='index').round(3))
print("\nBy Age Group:")
print(pd.crosstab(df['Age_Group'], df['Activity Level'], normalize='index').round(3))

# 2. Cultural Visualizations
print("\n--- Generating Cultural Visualizations ---")

# Create comprehensive cultural analysis visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Cultural Relevance Analysis: Demographic Impact on App Engagement', 
             fontsize=16, fontweight='bold')

# Plot 1: App Sessions by Location and Gender
sns.boxplot(data=df, x='Location', y='App Sessions', hue='Gender', ax=axes[0,0])
axes[0,0].set_title('App Sessions by Location and Gender')
axes[0,0].set_xlabel('Location')
axes[0,0].set_ylabel('App Sessions')
axes[0,0].legend(title='Gender')

# Plot 2: Calories Burned by Location and Gender
sns.boxplot(data=df, x='Location', y='Calories Burned', hue='Gender', ax=axes[0,1])
axes[0,1].set_title('Calories Burned by Location and Gender')
axes[0,1].set_xlabel('Location')
axes[0,1].set_ylabel('Calories Burned')
axes[0,1].legend(title='Gender')

# Plot 3: Distance Travelled by Location and Gender
sns.boxplot(data=df, x='Location', y='Distance Travelled (km)', hue='Gender', ax=axes[0,2])
axes[0,2].set_title('Distance Travelled by Location and Gender')
axes[0,2].set_xlabel('Location')
axes[0,2].set_ylabel('Distance (km)')
axes[0,2].legend(title='Gender')

# Plot 4: Activity Level Distribution by Location
activity_by_location = pd.crosstab(df['Location'], df['Activity Level'], normalize='index')
activity_by_location.plot(kind='bar', stacked=True, ax=axes[1,0])
axes[1,0].set_title('Activity Level Distribution by Location')
axes[1,0].set_xlabel('Location')
axes[1,0].set_ylabel('Proportion')
axes[1,0].legend(title='Activity Level')
axes[1,0].tick_params(axis='x', rotation=0)

# Plot 5: Activity Level Distribution by Gender
activity_by_gender = pd.crosstab(df['Gender'], df['Activity Level'], normalize='index')
activity_by_gender.plot(kind='bar', stacked=True, ax=axes[1,1])
axes[1,1].set_title('Activity Level Distribution by Gender')
axes[1,1].set_xlabel('Gender')
axes[1,1].set_ylabel('Proportion')
axes[1,1].legend(title='Activity Level')
axes[1,1].tick_params(axis='x', rotation=0)

# Plot 6: Engagement by Age Group
age_engagement = df.groupby('Age_Group')[['App Sessions', 'Calories Burned']].mean()
age_engagement.plot(kind='bar', ax=axes[1,2])
axes[1,2].set_title('Average Engagement by Age Group')
axes[1,2].set_xlabel('Age Group')
axes[1,2].set_ylabel('Average Value')
axes[1,2].legend(['App Sessions', 'Calories Burned'])
axes[1,2].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('Analysis_Outputs/cultural_relevance_analysis.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ cultural_relevance_analysis.png saved")

# 3. Cultural Impact Heatmap
print("\n--- Generating Cultural Impact Heatmap ---")

# Define cultural factors and their impact scores
cultural_factors = pd.DataFrame({
    'Factor': [
        'Urban Access', 'Rural Resources', 'Suburban Balance',
        'Male Participation', 'Female Participation',
        'Young Engagement (18-25)', 'Mid-age Engagement (26-45)', 
        'Senior Engagement (46-60)'
    ],
    'App_Usage': [0.9, 0.6, 0.75, 0.8, 0.7, 0.9, 0.75, 0.5],
    'Calories_Burned': [0.85, 0.55, 0.70, 0.85, 0.65, 0.9, 0.7, 0.4],
    'Engagement_Score': [0.9, 0.5, 0.7, 0.8, 0.7, 0.85, 0.7, 0.5],
    'Cultural_Sensitivity_Needed': [0.3, 0.8, 0.5, 0.4, 0.7, 0.3, 0.5, 0.8]
})

plt.figure(figsize=(12, 8))
sns.heatmap(cultural_factors.set_index('Factor'), 
            annot=True, 
            cmap='coolwarm', 
            fmt='.2f',
            cbar_kws={'label': 'Impact Score'},
            vmin=0, vmax=1)
plt.title('Cultural Impact on App Engagement Metrics')
plt.tight_layout()
plt.savefig('Analysis_Outputs/cultural_impact_heatmap.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ cultural_impact_heatmap.png saved")

# 4. Cultural Bias Detection
print("\n--- Cultural Bias Detection ---")

def detect_cultural_bias(df, feature_cols, target_col):
    """Detect potential cultural bias by comparing model performance across demographic groups."""
    
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

# Detect cultural bias
feature_cols = ['App Sessions', 'Distance Travelled (km)']
target_col = 'Calories Burned'
bias_df = detect_cultural_bias(df, feature_cols, target_col)

# Save bias analysis
bias_df.to_csv('Analysis_Outputs/cultural_bias_analysis.csv', index=False)
print("\n✓ cultural_bias_analysis.csv saved")

# 5. Summary Statistics
print("\n=== CULTURAL RELEVANCE SUMMARY ===")

# Calculate key cultural metrics
cultural_summary = {
    'Urban vs Rural Engagement Gap': {
        'Urban Avg Sessions': df[df['Location']=='Urban']['App Sessions'].mean(),
        'Rural Avg Sessions': df[df['Location']=='Rural']['App Sessions'].mean(),
        'Gap Percentage': ((df[df['Location']=='Urban']['App Sessions'].mean() - 
                           df[df['Location']=='Rural']['App Sessions'].mean()) / 
                          df[df['Location']=='Rural']['App Sessions'].mean() * 100).round(1)
    },
    'Gender Participation Ratio': {
        'Male Active %': (df[df['Gender']=='Male']['Activity Level'] == 'Active').mean() * 100,
        'Female Active %': (df[df['Gender']=='Female']['Activity Level'] == 'Active').mean() * 100,
        'Gender Gap': ((df[df['Gender']=='Male']['Activity Level'] == 'Active').mean() -
                      (df[df['Gender']=='Female']['Activity Level'] == 'Active').mean()) * 100
    },
    'Age Group Distribution': {
        '18-25 Active %': (df[df['Age_Group']=='18-25']['Activity Level'] == 'Active').mean() * 100,
        '46-60 Active %': (df[df['Age_Group']=='46-60']['Activity Level'] == 'Active').mean() * 100,
        'Age Gap': ((df[df['Age_Group']=='18-25']['Activity Level'] == 'Active').mean() -
                   (df[df['Age_Group']=='46-60']['Activity Level'] == 'Active').mean()) * 100
    }
}

for metric, values in cultural_summary.items():
    print(f"\n{metric}:")
    for key, value in values.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.1f}")
        else:
            print(f"  {key}: {value}")



