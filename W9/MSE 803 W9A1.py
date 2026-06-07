import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

file_path = os.path.join(os.path.dirname(__file__), 'Fitness_App_User_Data.xlsx')

def process_fitness_data(file_path):
    # Load the data
    df = pd.read_excel(file_path)
    
    # 1. Enforce Data Types
    print("Enforcing data types...")
    required_types = {
        'User_ID': 'int64',
        'Age': 'int64',
        'Gender': 'object',
        'Workouts_per_Week': 'int64',
        'Avg_Session_Duration_Min': 'float64',
        'Steps_per_Day': 'int64',
        'Subscription_Type': 'object',
        'Churned': 'bool'
    }
    
    for col, dtype in required_types.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    # 2. Check for users with same data but different IDs
    # We define columns to check (all except User_ID)
    cols_to_check = [c for c in df.columns if c != 'User_ID']
    duplicates = df[df.duplicated(subset=cols_to_check, keep=False)]
    
    if not duplicates.empty:
        print("\n--- Potential Duplicate Users Found (Same data, different IDs) ---")
        print(duplicates)
        # Here you could add logic to drop them or keep one, if desired
    else:
        print("\nNo duplicate users with identical data found.")

    # 3. Column Summary
    summary = pd.DataFrame({
        'Data Type': df.dtypes,
        'Value Count': df.count(),
        'Missing Values': df.isnull().sum()
    })
    print("\nColumn Summary:")
    print(summary)
    
    # 4. Detect and handle outliers
    print("\nChecking for potential outliers (Numerical columns):")
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        if not outliers.empty:
            print(f"- {col}: {len(outliers)} potential outliers found.")
            for index, row in outliers.iterrows():
                decision = input(f"\nOutlier at index {index} (value: {row[col]}). Do you want to (K)eep or (C)hange it? ").lower()
                if decision == 'c':
                    new_value = input(f"Enter the new value for index {index}: ")
                    try:
                        df.at[index, col] = float(new_value)
                    except ValueError:
                        df.at[index, col] = new_value
                    print(f"Value updated.")

    # 5. Save the cleaned file
    output_path = os.path.join(os.path.dirname(__file__), 'Fitness_App_User_Data_Cleaned.xlsx')
    df.to_excel(output_path, index=False)
    print(f"\nProcessing complete. File saved as: {output_path}")
    return df

def perform_clustering_analysis(df):
    # 1. Select appropriate features (Numerical data only)
    features = ['Age', 'Workouts_per_Week', 'Avg_Session_Duration_Min', 'Steps_per_Day']
    x = df[features].dropna()
    
    # 2. Scale the data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    # 3. Determine optimal clusters using the Elbow Method
    print("\n--- Clustering Analysis: Elbow Method ---")
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(x_scaled)
        inertia.append(kmeans.inertia_)
        
    plt.plot(k_range, inertia, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.show()

    print("Elbow Method complete. Recommend picking K where the drop in inertia slows down.")
    
    suggested_k = 3
    print(f"\nThe Elbow Method suggests looking for the 'elbow' in the plot.")
    user_k = input(f"We suggest k={suggested_k} clusters. Do you want to use this value? (Y/n): ").lower()
    
    if user_k == 'n':
        final_k = int(input("Please enter your desired number of clusters (k): "))
    else:
        final_k = suggested_k

    # 4. Apply K-Means 
    k = final_k
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_cluster = x.copy()
    df_cluster['Cluster'] = kmeans.fit_predict(x_scaled)
    
    # 5. Interpret the characteristics
    print(f"\n--- Cluster Characteristics (Means for {k} clusters) ---")
    print(df_cluster.groupby('Cluster').mean())
    
    return df_cluster
  
  
def perform_cluster_analytics(df_cluster, df_original):
    """
    Analyzes the composition of each cluster regarding Gender and Subscription Type.
    """
    print("\n--- Performing Cluster Composition Analytics ---")
    
    # 1. Integrate original categorical data back to the cluster results using index
    df_cluster['Gender'] = df_original.loc[df_cluster.index, 'Gender']
    df_cluster['Subscription_Type'] = df_original.loc[df_cluster.index, 'Subscription_Type']
    
    # 2. Gender distribution analysis (percentage per cluster)
    print("\n[Gender Distribution per Cluster (%)]")
    gender_dist = pd.crosstab(df_cluster['Cluster'], df_cluster['Gender'], normalize='index') * 100
    print(gender_dist.round(2).astype(str) + '%')
    
    # 3. Subscription distribution analysis (percentage per cluster)
    print("\n[Subscription Distribution per Cluster (%)]")
    sub_dist = pd.crosstab(df_cluster['Cluster'], df_cluster['Subscription_Type'], normalize='index') * 100
    print(sub_dist.round(2).astype(str) + '%')

    # 4. Churn Rate Analysis
    df_cluster['Churned'] = df_original.loc[df_cluster.index, 'Churned']
    
    print("\n[Churn Rate per Cluster (%)]")
    # Group by cluster and calculate the mean (True=1, False=0), then convert to percentage
    churn_rate = df_cluster.groupby('Cluster')['Churned'].mean() * 100
    print(churn_rate.round(2).astype(str) + '%')


# Run the Clean function
df_cleaned = process_fitness_data(file_path)

# Run the Clustering Analysis
cluster_results = perform_clustering_analysis(df_cleaned)

# Run the new Composition Analytics (Crucial: pass the cluster results AND the cleaned dataframe)
perform_cluster_analytics(cluster_results, df_cleaned)
