import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Current date for Tenure calculation (April 2026)
CURRENT_DATE = datetime(2026, 4, 27)

def validate_numeric(row, field_name, target_type):
    """Displays full row context and asks for manual numeric correction."""
    print("\n" + "="*40)
    print(f"MANUAL CORRECTION REQUIRED: {field_name.upper()}")
    print("="*40)
    print(f"Full Row Context:\n{row.to_frame().T}")
    print(f"Invalid Value: '{row[field_name]}'")
    
    while True:
        new_val = input(f"Enter the correct {field_name} ({target_type.__name__}): ")
        try:
            return target_type(new_val)
        except ValueError:
            print(f"Invalid input. Please enter a valid {target_type.__name__}.")

def validate_date(row):
    """Displays full row context and asks for manual date correction."""
    print("\n" + "="*40)
    print("MANUAL CORRECTION REQUIRED: JOIN DATE")
    print("="*40)
    print(f"Full Row Context:\n{row.to_frame().T}")
    print(f"Invalid Date: '{row['Join Date']}'")
    
    while True:
        new_date = input("Enter the correct date (dd/mm/yyyy): ")
        try:
            return datetime.strptime(new_date, '%d/%m/%Y')
        except ValueError:
            print("Incorrect format. Please use dd/mm/yyyy.")

# 1. Load Data
df = pd.read_csv('messy_dataset_Mukesh.csv')

# 2. Fix missing IDs
for i in range(len(df)):
    if pd.isna(df.loc[i, 'ID']):
        next_id = df['ID'].max() + 1 if not pd.isna(df['ID'].max()) else len(df) + 1
        df.loc[i, 'ID'] = next_id

# Ensure ID is numeric
df['ID'] = pd.to_numeric(df['ID'], errors='coerce')

# 3. Consolidate duplicates
# Rule: If ID is the same + 2 other traits, merge. 
# We use ffill/bfill to merge available data within ID groups.
df = df.groupby('ID').apply(lambda x: x.ffill().bfill().drop_duplicates())

# Reset_index and ensure ID becomes a column
df = df.reset_index(drop=False)  # Keep ID when resetting

# Check if ID is still there, if not reconstruct it
if 'ID' not in df.columns:
    if 'level_0' in df.columns:
        df = df.rename(columns={'level_0': 'ID'})
    else:
        # Reconstruct ID from index if needed
        df = df.reset_index()
        if 'ID' not in df.columns and 'level_0' in df.columns:
            df = df.rename(columns={'level_0': 'ID'})

# Convert ID to int
df['ID'] = df['ID'].astype(float).astype(int)

# 4. Set flexible types for the cleaning loop
df['Age'] = df['Age'].astype(object)
df['Salary'] = df['Salary'].astype(object)
df['Join Date'] = df['Join Date'].astype(object)

# 5. Interactive Cleaning Loop
for idx, row in df.iterrows():
    # Validate Age (Int)
    try:
        val = str(row['Age'])
        if pd.isna(row['Age']) or val.lower() == 'nan':
            df.at[idx, 'Age'] = np.nan
        else:
            df.at[idx, 'Age'] = int(float(val))
    except:
        df.at[idx, 'Age'] = validate_numeric(row, 'Age', int)

    # Validate Salary (Float)
    try:
        val = str(row['Salary'])
        if pd.isna(row['Salary']) or val.lower() == 'nan':
            df.at[idx, 'Salary'] = np.nan
        else:
            df.at[idx, 'Salary'] = float(val.replace(',', ''))
    except:
        df.at[idx, 'Salary'] = validate_numeric(row, 'Salary', float)

    # Validate Join Date (dd/mm/yyyy)
    try:
        date_val = str(row['Join Date'])
        if pd.isna(row['Join Date']) or date_val.lower() == 'nan':
            df.at[idx, 'Join Date'] = np.nan
        else:
            df.at[idx, 'Join Date'] = datetime.strptime(date_val, '%d/%m/%Y')
    except ValueError:
        df.at[idx, 'Join Date'] = validate_date(row)

# 6. Save Cleaned Dataset
df_save = df.copy()
df_save['Join Date'] = df_save['Join Date'].apply(
    lambda x: x.strftime('%d/%m/%Y') if isinstance(x, datetime) else "N/A"
)
df_save = df_save.fillna("N/A")
df_save.to_csv('clean_dataset_Mukesh.csv', index=False)
print("\nSuccess! 'clean_dataset_Mukesh.csv' has been generated.")

# --- 7. HEATMAP & TENURE ANALYSIS ---

df_analysis = df.copy()
df_analysis['Age'] = pd.to_numeric(df_analysis['Age'], errors='coerce')
df_analysis['Salary'] = pd.to_numeric(df_analysis['Salary'], errors='coerce')

# Calculate Tenure (Years since joining)
def get_tenure(dt):
    if isinstance(dt, datetime):
        return (CURRENT_DATE - dt).days / 365.25
    return np.nan

df_analysis['Tenure'] = df_analysis['Join Date'].apply(get_tenure)

# Filter rows where Age, Salary, and Tenure exist for the correlation
df_corr = df_analysis[['Age', 'Salary', 'Tenure']].dropna()

if not df_corr.empty:
    plt.figure(figsize=(10, 8))
    # Pearson algorithm measures linear correlation
    sns.heatmap(df_corr.corr(method='pearson'), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Pearson Correlation Heatmap: Age, Salary, & Tenure")
    plt.savefig('correlation_heatmap.png')
    plt.show()

# Outlier Detection (Salary)
if not df_analysis['Salary'].isna().all():
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df_analysis['Salary'])
    plt.title("Salary Outliers Identification (Boxplot)")
    plt.savefig('salary_outliers.png') # Salva imagem
    print("Outlier boxplot saved as 'salary_outliers.png'.")
    plt.show()

    # Console Outlier Report
    q1, q3 = df_analysis['Salary'].quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = df_analysis[(df_analysis['Salary'] < (q1 - 1.5 * iqr)) | (df_analysis['Salary'] > (q3 + 1.5 * iqr))]
    
    if not outliers.empty:
        print("\n--- OUTLIER REPORT ---")
        print(outliers[['ID', 'Name', 'Salary']])