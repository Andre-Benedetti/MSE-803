import pandas as pd
import matplotlib.pyplot as plt

def perform_analysis(file_path):
    # 1. Load the dataset
    try:
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip().str.lower()
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Group and Aggregate 
    story_data = df.groupby('category').agg(
        Sales_Sum=('sales_sum', 'sum'),
        Sales_Mean=('sales_mean', 'mean'),
        Sales_Count=('sales_count', 'sum'),
        Quantity_Sum=('quantity_sum', 'sum'),
        Quantity_Mean=('quantity_mean', 'mean')
    ).round(2)

    # Rename for display
    story_data.columns = ['Sales Sum', 'Sales Mean', 'Sales Count', 'Quantity Sum', 'Quantity Mean']
    story_data.index.name = 'Category'

    # 3. Create the Table Image (Final Outcome)
    fig, ax = plt.subplots(figsize=(10, 2)) # Adjust size to fit the table
    ax.axis('off') # Hide axes

    # Create the table object
    the_table = ax.table(
        cellText=story_data.values, 
        colLabels=story_data.columns, 
        rowLabels=story_data.index,
        loc='center',
        cellLoc='center'
    )

    
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1.2, 1.8) # Adjust scaling for better padding

    
    for (row, col), cell in the_table.get_celld().items():
        if row == 0 or col == -1:
            cell.set_text_props(weight='bold')

    # 4. Save the table as the final outcome image
    plt.savefig('final_outcome.png', bbox_inches='tight', dpi=300)
    print("Table image saved as 'final_outcome.png'.")

if __name__ == "__main__":
    file_name = 'Data_set_w1A1.xlsx' 
    perform_analysis(file_name)