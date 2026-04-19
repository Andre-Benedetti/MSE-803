import pandas as pd
import glob
import os

# 1. Setup path and search for all CSV files in the directory
path = './'  # Current directory
all_files = glob.glob(os.path.join(path, "*.csv"))

# Check if any CSV files exist to avoid errors
if not all_files:
    print("No CSV files found in the directory.")
    exit()

print(f"Found {len(all_files)} CSV files. Starting structure validation...")

df_list = []
reference_columns = None  # This will store the expected column layout

# 2. Load and Validate each file
for filename in all_files:
    try:
        # Load the current file into a temporary DataFrame
        temp_df = pd.read_csv(filename)
        
        # Capture column names as a list to check order and naming
        current_columns = temp_df.columns.tolist()

        if reference_columns is None:
            # The first file found sets the standard for all others
            reference_columns = current_columns
            print(f"Standard structure set by: {filename}")
            df_list.append(temp_df)
        else:
            # Compare current file structure against the reference
            if current_columns == reference_columns:
                df_list.append(temp_df)
                print(f"Validated: {filename}")
            else:
                # Flag files that do not match the expected format
                print(f"CRITICAL ERROR: {filename} has a different structure (names or order)!")
                
    except Exception as e:
        print(f"Error loading {filename}: {e}")

# 3. Merge files only if all structures are consistent
if len(df_list) == len(all_files):
    # Concatenate all validated DataFrames into a single one
    df = pd.concat(df_list, ignore_index=True)
    print("\nAll datasets matched and successfully merged!\n")
else:
    print(f"\nWarning: Only {len(df_list)} out of {len(all_files)} files were consistent.")
    user_input = input("Do you want to merge only the consistent files? (y/n): ")
    if user_input.lower() == 'y':
        df = pd.concat(df_list, ignore_index=True)
    else:
        print("Operation cancelled by user.")
        exit()

# 4. Final Output and Data Inspection
print("--- First 5 rows of the combined dataset ---")
print(df.head())

print("\n--- Column names and Data types ---")
# .info() is used to verify Dtypes and check for missing values
print(df.info()) 

# Display final row and column count
rows, columns = df.shape
print(f"\n--- Combined Dataset Dimensions ---")
print(f"Total rows: {rows}")
print(f"Total columns: {columns}")