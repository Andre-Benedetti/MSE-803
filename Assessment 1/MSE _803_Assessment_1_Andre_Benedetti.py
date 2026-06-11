import pandas as pd
import os
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns 

class DataProcessor:
    def __init__(self, file_path):
        """Initialize the processor with the path to the Excel file."""
        self.file_path = file_path
        self.df = None

    def load_and_merge_data(self):
        """Loads data from the Excel file reading tables from the same sheet."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} was not found.")
        
        # We read the same sheet but slice the columns for each table
        # usecols: defines which columns to read (e.g., "A:D" and "F:J")
        # header: tells pandas which row contains the titles
        
        # Adjust these ranges according to our actual Excel layout
        df_water = pd.read_excel(self.file_path, sheet_name=0, header=1, usecols="A:E")
        df_fish = pd.read_excel(self.file_path, sheet_name=0, header=1, usecols="G:K")

        # Debugging: check what columns are actually being read
        print("Water columns:", df_water.columns)
        print("Fish columns:", df_fish.columns)

        # Rename the columns in df_fish to match df_water so they can be merged
        # We target the specific columns that were renamed to .1
        df_fish = df_fish.rename(columns={'Site ID.1': 'Site ID', 'Date.1': 'Date'})
        
        # Clean potential empty rows that might be imported
        df_water = df_water.dropna(how='all')
        df_fish = df_fish.dropna(how='all')
        
        # Merge based on Site ID and Date
        self.df = pd.merge(df_water, df_fish, on=['Site ID', 'Date'], how='inner')
        print("Data loaded and merged successfully from the same sheet.")
        

    def inspect_data(self):
        """Displays data types, missing values, and duplicates."""
        print("\n--- Data Types ---")
        print(self.df.dtypes)
        print("\n--- Missing Values ---")
        print(self.df.isnull().sum())
        print(f"\n--- Duplicates: {self.df.duplicated().sum()} ---")

    def clean_outliers(self):
        """Detects outliers (Z-score > 3) and allows manual correction."""
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            mean = self.df[col].mean()
            std = self.df[col].std()
            
            # Identify outliers
            outlier_indices = self.df[(self.df[col] - mean).abs() > 3 * std].index
            
            for idx in outlier_indices:
                print(f"\nOutlier in '{col}' at index {idx}: {self.df.at[idx, col]}")
                choice = input("Change this value? (y/n): ").lower()
                if choice == 'y':
                    new_val = float(input(f"Enter new value for {col}: "))
                    self.df.at[idx, col] = new_val
                    print("Value updated.")
    
    def clean_outliers(self):
            """Detects outliers using Z-score (> 3 std) and allows manual correction."""
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            found_outliers = False  # Track if any outlier is found
            
            for col in numeric_cols:
                mean = self.df[col].mean()
                std = self.df[col].std()
                
                # Identify outliers where Z-score > 3
                outlier_indices = self.df[(self.df[col] - mean).abs() > 3 * std].index
                
                for idx in outlier_indices:
                    found_outliers = True
                    print(f"\n[!] Outlier detected in '{col}' at index {idx}: {self.df.at[idx, col]}")
                    # Shows the full row for context
                    print(self.df.loc[idx].to_frame().T) 
                    
                    choice = input("Do you want to change this value? (y/n): ").lower()
                    if choice == 'y':
                        new_val = float(input(f"Enter the new value for {col}: "))
                        self.df.at[idx, col] = new_val
                        print("Value updated.")
            
            if not found_outliers:
                print("\n--- Outlier Check ---")
                print("No outliers were detected in the numeric columns.")

    def save_to_csv(self, output_name):
        """Saves the processed dataframe to a CSV file."""
        self.df.to_csv(output_name, index=False)
        print(f"\nData saved to {output_name}.")

class DataAnalysis:
    def __init__(self, dataframe):
        self.df = dataframe
        self.predictors = ['Temperature (°C)', 'pH', 'Dissolved Oxygen (mg/L)']
    
    def analyze_site_variation(self):
        cols_to_plot = ['Count', 'Avg. Size (cm)', 'Temperature (°C)', 'pH', 'Dissolved Oxygen (mg/L)']
        site_variation = self.df.groupby('Site ID')[cols_to_plot].mean()
        df_plot = site_variation.T 
        
        ax = df_plot.plot(kind='bar', figsize=(12, 7), rot=0)
        plt.title("Comparison of Environmental and Biological Metrics by Site")
        plt.ylabel("Mean Value")
        plt.xlabel("Metric Category")
        plt.legend(title="Monitoring Site", loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("spatial_variation.png") 
        plt.close() 

    def perform_correlation(self):
        targets = ['Count', 'Avg. Size (cm)']
        for target in targets:
            numeric_df = self.df[self.predictors + [target]]
            corr_matrix = numeric_df.corr(method='pearson')
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title(f"Correlation Matrix: Water Quality vs {target}")
            plt.tight_layout()
            plt.savefig(f"correlation_{target.replace(' ', '_')}.png") 
            plt.close() 
            
            print(f"\n--- Detailed Correlation Matrix ({target}) ---")
            print(corr_matrix)

    def perform_mlr_and_visualise(self):
        targets = ['Count', 'Avg. Size (cm)']
        for target in targets:
            X = self.df[self.predictors]
            X = sm.add_constant(X)
            y = self.df[target]
            model = sm.OLS(y, X).fit()
            print(f"\n--- Multiple Linear Regression Results: {target} ---")
            print(model.summary())
                       

if __name__ == "__main__":
    processor = DataProcessor('Data_Set_Assignmnet_1.xlsx')
    try:
        processor.load_and_merge_data()
        processor.inspect_data()
        processor.clean_outliers()
        processor.save_to_csv('verified_data.csv')

        analyzer = DataAnalysis(processor.df)
        analyzer.perform_correlation()
        analyzer.perform_mlr_and_visualise()
        analyzer.analyze_site_variation()
            
    except Exception as e:
        print(f"An error occurred: {e}")