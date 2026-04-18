import pandas as pd
import glob
import os

# 1 read all datasets and combine them into one dataframe
# Define the pattern to find all csv files
path = './'  # Current directory
all_files = glob.glob(os.path.join(path, "*.csv"))

# List to store individual dataframes
df_list = []

print(f"Found {len(all_files)} CSV files. Loading data...")

# 1. Load all datasets
for filename in all_files:
    try:
        # Reading each file and appending to the list
        temp_df = pd.read_csv(filename)
        df_list.append(temp_df)
        print(f"Successfully loaded: {filename}")
    except Exception as e:
        print(f"Error loading {filename}: {e}")

# Combine all dataframes into one
if df_list:
    df = pd.concat(df_list, ignore_index=True)
    print("\nAll datasets successfully merged!\n")
else:
    print("No CSV files found.")
    exit()

# 2. Display the first 5 rows
print("--- First 5 rows of the dataset ---")
print(df.head())
print("\n")

# 3. Identify column names and data types
print("--- Column names and Data types ---")
# .info() provides a concise summary of the dataframe
print(df.info()) 
print("\n")

# 4. Count total rows and columns
# df.shape returns a tuple (rows, columns)
rows, columns = df.shape
print(f"--- Dataset Dimensions ---")
print(f"Total rows: {rows}")
print(f"Total columns: {columns}")