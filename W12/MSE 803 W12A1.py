import pandas as pd
import numpy as np

def detect_and_correct_data(file_path):
    # 1. Load the dataset
    try:
        df = pd.read_csv(file_path)
        print(f"File '{file_path}' successfully loaded.")
        print(f"Total records: {len(df)} rows.\n")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found in the current directory.")
        return

    # 2. Check for Missing Values
    print("Checking for missing values...")
    missing_total = df.isnull().sum().sum()
    rows_with_nulls = df[df.isnull().any(axis=1)].index.tolist()
    
    if missing_total == 0:
        print("-> No missing values found.")
    else:
        print(f"-> Found {len(rows_with_nulls)} rows with missing values.")

    # 3. Check for Outliers based on Custom Business Rules
    print("\nChecking for outliers using specific business rules...")
    
    rows_with_outliers = set()
    outliers_dict = {} # Stores which specific columns triggered the outlier flag

    # Helper function to track outliers per row
    def add_outlier(row_idx, issue_description):
        rows_with_outliers.add(row_idx)
        if row_idx not in outliers_dict:
            outliers_dict[row_idx] = []
        outliers_dict[row_idx].append(issue_description)

    # Iterate through the dataframe to apply rules line by line
    for idx, row in df.iterrows():
        # Rule 1: total_price must equal (unit_price * quantity) + tax
        # We use a small tolerance (0.01) to account for float rounding differences
        expected_total = (row['unit_price'] * row['quantity']) + row['tax']
        if abs(row['total_price'] - expected_total) > 0.01:
            add_outlier(idx, f"total_price formula mismatch (Expected: {expected_total:.2f}, Found: {row['total_price']})")
            
        # Rule 2: reward_points is an outlier if it is greater than 10% of total_price
        if row['reward_points'] > (0.10 * row['total_price']):
            add_outlier(idx, f"reward_points too high (> 10% of total_price: {row['reward_points']})")
            
        # Rule 3: tax is an outlier if it is greater than 8% or less than 6% of total_price
        # Prevent division by zero if total_price happens to be 0
        if row['total_price'] > 0:
            tax_percentage = row['tax'] / row['total_price']
            if tax_percentage > 0.08 or tax_percentage < 0.06:
                add_outlier(idx, f"tax rate out of bounds ({tax_percentage*100:.2f}% of total_price)")

    print(f"-> Found {len(rows_with_outliers)} rows violating the business rules.\n")

    # Combine all unique row indices that require user review
    all_suspicious_rows = list(set(rows_with_nulls).union(rows_with_outliers))
    all_suspicious_rows.sort()

    if not all_suspicious_rows:
        print("Your dataset is clean. No business rule violations or missing values found.")
        return df

    print(f"A total of {len(all_suspicious_rows)} rows require your review.")
    print("-" * 60)

    # 4. Interactive loop for user input
    for idx in all_suspicious_rows:
        print(f"\n[ROW {idx}] requires attention.")
        
        # Display the detection trigger reason
        reasons = []
        if idx in rows_with_nulls:
            null_cols = df.columns[df.iloc[idx].isnull()].tolist()
            reasons.append(f"Missing values in columns: {null_cols}")
        if idx in outliers_dict:
            reasons.extend(outliers_dict[idx])
        
        print("Reasons:")
        for r in reasons:
            print(f"  - {r}")
            
        print("\nCurrent row data:")
        # Display every column value for the current row
        row_data = df.iloc[idx]
        for col in df.columns:
            print(f"  {col}: {row_data[col]}")
        
        # Present options to the user
        print("\nWhat would you like to do?")
        print("[1] Edit a column value")
        print("[2] Keep current data and skip to the next row")
        print("[3] Save changes and exit program")
        
        choice = input("Select an option (1, 2, or 3): ").strip()
        
        if choice == '3':
            print("\nExiting interactive mode...")
            break
            
        elif choice == '1':
            while True:
                col_to_edit = input("\nEnter the EXACT name of the column you want to edit (or type 'back'): ").strip()
                if col_to_edit.lower() == 'back':
                    break
                
                if col_to_edit in df.columns:
                    new_value = input(f"Enter the new value for '{col_to_edit}': ")
                    
                    # Safe type casting based on the column's original data type
                    col_type = df[col_to_edit].dtype
                    try:
                        if np.issubdtype(col_type, np.integer):
                            new_value = int(new_value)
                        elif np.issubdtype(col_type, np.floating):
                            new_value = float(new_value)
                        
                        # Apply change to the DataFrame
                        df.at[idx, col_to_edit] = new_value
                        print(f"Column '{col_to_edit}' updated successfully.")
                        
                        # Ask if the user wants to update another column in this specific row
                        another = input("Do you want to edit another column in this same row? (y/n): ").strip().lower()
                        if another != 'y':
                            break
                    except ValueError:
                        print(f"Error: The entered value cannot be converted to the required column type ({col_type}). Please try again.")
                else:
                    print("Error: Column not found. Check the spelling and case-sensitivity.")
                    
        elif choice == '2':
            print("Skipped. Moving to the next record...")
            continue
        else:
            print("Invalid option. Skipping row for safety.")

    # 5. Save the updated DataFrame
    save_changes = input("\nDo you want to save your changes to a new CSV file? (y/n): ").strip().lower()
    if save_changes == 'y':
        output_name = "sales_cleaned.csv"
        df.to_csv(output_name, index=False)
        print(f"Changes successfully saved to '{output_name}'.")
    else:
        print("Program closed without saving changes.")
    
    return (df)



def run_sales_analysis(df):
    print("\n" + "="*20 + " DATA ANALYSIS REPORT " + "="*20)
    
    # Analysis 1: Which factor (branch, customer type, gender) most influences total sales
    print("\n1. INFLUENCE ON TOTAL SALES (total_price)")
    factors = ['branch', 'customer_type', 'gender']
    
    for factor in factors:
        grouped = df.groupby(factor)['total_price'].agg(['count', 'mean', 'sum']).reset_index()
        print(f"\nGrouped by {factor.upper()}:")
        for _, row in grouped.iterrows():
            print(f"  {row[factor]}: Total Sales = {row['sum']:.2f}, Average Ticket = {row['mean']:.2f}, Transactions = {row['count']}")
            
    # Analysis 2: Product category generating the most sales per branch
    print("\n" + "-"*60)
    print("2. HIGHEST SELLING PRODUCT CATEGORIES PER BRANCH")
    pivot_sales = df.pivot_table(index='product_category', columns='branch', values='total_price', aggfunc='sum')
    
    for col in pivot_sales.columns:
        top_category = pivot_sales[col].idxmax()
        top_value = pivot_sales[col].max()
        print(f"  Branch {col}: Best category is '{top_category}' with a total of {top_value:.2f}")
    
    # Analysis 3: Which factor (branch, customer type, gender) most influences product category selection
    print("\n" + "-"*60)
    print("3. INFLUENCE OF FACTORS ON PRODUCT CATEGORY (Preference %)")
    
    for factor in factors:
        print(f"\nDistribution by {factor.upper()} (Percentage of purchases within each group):")
        crosstab = pd.crosstab(df['product_category'], df[factor], normalize='columns') * 100
        print(crosstab.round(2).to_string())


# Run the cleaning pipeline
if __name__ == "__main__":
    cleaned_df = detect_and_correct_data("sales.csv")
    if cleaned_df is not None:
        run_sales_analysis(cleaned_df)