import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sqlite3

# ==========================================
# Activity 1: DATA ANALYSIS
# ==========================================
def perform_data_analysis(file_path):
    """
    Handles data cleaning, outlier detection, 
    and saves visualizations.
    """
    try:
        # 1. Loading
        df = pd.read_csv(file_path)
        
        print("--- Activity 1: DATA ANALYSIS ---")
        print(f"Dataset loaded: {df.shape[0]} rows and {df.shape[1]} columns.")
        
        # 2. Structural Inspection & Types
        print("\n[Column Types]")
        print(df.dtypes)
        
        # 3. Data Cleaning (Missing Values)
        print("\n[Missing Values Report]")
        missing = df.isnull().sum()
        print(missing[missing > 0] if missing.any() else "No missing values found.")

        # 4. Outlier Detection (IQR Method)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print("\n[Outlier Detection Log]")
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower) | (df[col] > upper)]
            if not outliers.empty:
                print(f"Variable '{col}': {len(outliers)} outliers detected.")

        # 5. Visualizations
        sns.set_theme(style="whitegrid")
        
        # Boxplot for Outliers
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df[numeric_cols])
        plt.title("Outlier Distribution (Boxplot)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('outlier_boxplot.png')
        plt.close()
        print("\n- Saved: outlier_boxplot.png")

        # Correlation Heatmap
        target = next((c for c in df.columns if 'score' in c.lower()), df.columns[1])
        plt.figure(figsize=(12, 10))
        corr_matrix = df.select_dtypes(include=['number']).corr()
        target_corr = corr_matrix[[target]].sort_values(by=target, ascending=False)
        
        sns.heatmap(target_corr, annot=True, cmap='RdYlGn', fmt=".2f", linewidths=0.5)
        plt.title(f"Variables Correlation with {target}")
        plt.savefig('correlation_heatmap.png', bbox_inches='tight')
        plt.close()
        print("- Saved: correlation_heatmap.png")
        
        return df

    except Exception as e:
        print(f"Error in Analysis Object: {e}")
        return None

# ==========================================
# Activity 2: SQL QUERIES
# ==========================================
def execute_sql_queries(df):
    """
    Transforms data into a relational table and 
    executes complex SQL aggregations.
    """
    try:
        # Standardize names for SQL compatibility
        df_sql = df.copy()
        df_sql.columns = [c.replace(' ', '_').replace('.', '_') for c in df_sql.columns]
        
        # Mapping dynamic columns
        target = next((c for c in df_sql.columns if 'score' in c.lower()))
        gdp = next((c for c in df_sql.columns if 'gdp' in c.lower()))
        corruption = next((c for c in df_sql.columns if 'corruption' in c.lower()))

        # Connect to in-memory DB
        conn = sqlite3.connect(':memory:')
        df_sql.to_sql('happiness_table', conn, index=False)

        print("\n" + "="*40)
        print("--- Activity 2: SQL QUERIES ---")

        # PART 1: GDP Categories & Ranking
        query_1 = f"""
        WITH Tiers AS (
            SELECT Country, {gdp} as GDP, {target} as Score,
            NTILE(3) OVER (ORDER BY {gdp} ASC) as gdp_tile
            FROM happiness_table
        )
        SELECT 
            CASE 
                WHEN gdp_tile = 1 THEN 'Low GDP'
                WHEN gdp_tile = 2 THEN 'Medium GDP'
                ELSE 'High GDP' 
            END AS GDP_Category,
            Country, 
            Score,
            ROUND(AVG(Score) OVER (PARTITION BY gdp_tile), 3) as Avg_Happiness_Category,
            RANK() OVER (PARTITION BY gdp_tile ORDER BY Score DESC) as Rank_In_Category
        FROM Tiers
        ORDER BY gdp_tile DESC, Rank_In_Category ASC;
        """

        # PART 2: Corruption Perception Analysis
        query_2 = f"""
        SELECT 
            Corruption_Level, 
            COUNT(*) as Country_Count,
            ROUND(AVG({target}), 2) as Avg_Happiness,
            ROUND(AVG({gdp}), 2) as Avg_GDP
        FROM (
            SELECT {target}, {gdp},
            CASE 
                WHEN {corruption} > (SELECT AVG({corruption}) FROM happiness_table) 
                THEN 'High Corruption Perception' 
                ELSE 'Low Corruption Perception' 
            END AS Corruption_Level
            FROM happiness_table
        ) AS Subquery
        GROUP BY Corruption_Level;
        """

        # Print results
        print("\n[SQL Result 1: GDP Categories & Ranking]")
        res1 = pd.read_sql_query(query_1, conn)
        print(res1)

        print("\n[SQL Result 2: Corruption Perception Analysis]")
        res2 = pd.read_sql_query(query_2, conn)
        print(res2)

        conn.close()

    except Exception as e:
        print(f"Error in SQL Object: {e}")

if __name__ == "__main__":
    FILE = 'world_happiness_dataset.csv'
    
    # Run Data Analysis
    data_df = perform_data_analysis(FILE)
    
    # Run SQL Queries
    if data_df is not None:
        execute_sql_queries(data_df)