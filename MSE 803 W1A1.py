import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_analysis(file_path):
    """
    Loads an Excel file, performs data aggregation to uncover the 
    underlying business story, and exports a visual summary.
    """
    try:
        # Load the Excel dataset
        df = pd.read_excel(file_path)
        print("Dataset successfully loaded!")
    except Exception as e:
        print(f"Error loading the file: {e}")
        return

    story_data = df.groupby('Category').agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).sort_values(by='Sales', ascending=False)

    print("\n--- Aggregation Summary (The Data Story) ---")
    print(story_data)

    # Visualization: Creating the final outcome screenshot
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    # Primary axis: Bar plot for Total Sales
    ax = sns.barplot(
        x=story_data.index, 
        y=story_data['Sales'], 
        palette='viridis', 
        label='Total Sales'
    )
    
    # Secondary axis: Line plot for Profitability trends
    ax2 = ax.twinx()
    sns.lineplot(
        x=story_data.index, 
        y=story_data['Profit'], 
        marker='o', 
        color='crimson', 
        linewidth=2.5, 
        label='Total Profit', 
        ax=ax2
    )
    
    # Formatting titles and labels for a professional report look
    ax.set_title('Business Performance Analysis: Revenue vs. Profitability by Category', fontsize=14)
    ax.set_xlabel('Product Category', fontsize=12)
    ax.set_ylabel('Sales Volume ($)', fontsize=12)
    ax2.set_ylabel('Net Profit ($)', fontsize=12)
    
    plt.tight_layout()
    
    # Save the result as a screenshot for the GitHub README
    plt.savefig('final_outcome.png')
    print("\nAnalysis complete! 'final_outcome.png' has been generated.")

if __name__ == "__main__":
    file_name = 'Data_set_w1A1.xlsx' 
    perform_analysis(file_name)