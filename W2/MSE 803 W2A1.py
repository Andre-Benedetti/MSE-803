import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# 1. Setup path and search for all CSV files in the directory
path = './'  # Current directory
all_files = glob.glob(os.path.join(path, "*PRSA_Data_*.csv"))

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

# 5. Replace missing values ---
# We use the mean for numerical columns and the mode for categorical ones
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Fill categorical 'wd' (wind direction) with the most frequent value (mode)
df['wd'] = df['wd'].fillna(df['wd'].mode()[0])

# 6. Descriptive Statistics (Mean, Median, Min, Max, Std) ---
stats = df.describe().T
stats['median'] = df[num_cols].median()
print("\n--- Descriptive Statistics ---")
print(stats[['mean', 'median', 'std', 'min', 'max']])

# 7. Filter data by station ---
# Example: Filtering for 'Aotizhongxin' 
station_name = 'Aotizhongxin'
aotizhongxin_data = df[df['station'] == station_name]
print(f"\nFiltered data for {station_name}: {len(aotizhongxin_data)} rows.")

# 8. Calculate average pollution levels ---
pollution_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
avg_pollution = df.groupby('station')[pollution_cols].mean()
print("\n--- Average Pollution Levels per Station ---")
print(avg_pollution)

# 9. Visualizations ---
sns.set_theme(style="whitegrid")

# Histogram of PM2.5
plt.figure(figsize=(12, 7))
sns.histplot(df['PM2.5'], bins=100, kde=True, color='teal', alpha=0.4)
aqi_breaks = [0, 12, 35.4, 55.4, 150.4, 250.4] 
aqi_labels = ['Good', 'Moderate', 'Unhealthy (Sens.)', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
aqi_colors = ['#00E400', '#D4D400', '#FF7E00', '#FF0000', '#8F3F97', '#7E0023']

# Get the maximum Y-axis value to position text correctly
ymax = plt.gca().get_ylim()[1]
for i, threshold in enumerate(aqi_breaks):
    # Add vertical line (except for the 0 threshold)
    if threshold > 0:
        plt.axvline(threshold, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
    
    # Zig-Zag logic:
    # If index is even (0, 2, 4 -> 1st, 3rd, 5th label): Position higher (90% of ymax)
    # If index is odd (1, 3, 5 -> 2nd, 4th, 6th label): Position lower (80% of ymax)
    if i % 2 == 0:
        v_pos = ymax * 0.90
    else:
        v_pos = ymax * 0.80

    plt.text(threshold + 2, v_pos, aqi_labels[i], 
             color=aqi_colors[i], fontweight='bold', fontsize=9)

plt.title('PM2.5 Distribution', fontsize=16)
plt.xlabel(r'PM2.5 Concentration ($\mu g/m^3$)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xlim(0, 500)
plt.tight_layout()
plt.savefig('pm25_histogram.png', dpi=300)
plt.close()

# Line plot of PM2.5 over time
# Convert components to a single datetime column for plotting
df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
plt.figure(figsize=(15, 7))
plt.plot(df['datetime'], df['PM2.5'], color='darkred', linewidth=0.5, alpha=0.7)
plt.title('PM2.5 Concentration Over Time')

# Calculate and plot the Trend Line (30-day Moving Average)
# 24 hours * 30 days = 720 points window
df_sorted = df.sort_values('datetime') # Ensure data is chronological
trend = df_sorted['PM2.5'].rolling(window=720, center=True).mean()
plt.plot(df_sorted['datetime'], trend, color='black', linewidth=2, label='30-Day Trend (Moving Avg)')

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y\n%b'))
plt.xlabel('Year', fontsize=12)

plt.ylabel(r'PM2.5 Concentration ($\mu g/m^3$)', fontsize=12)

plt.xticks(rotation=0, ha='center')

plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('pm25_over_time.png', dpi=300)
plt.close()

# Boxplot of pollutants
plt.figure(figsize=(12, 6))
# CO is excluded because its scale (thousands) would squash the other variables
sns.boxplot(data=df[['PM2.5', 'PM10', 'SO2', 'NO2', 'O3']])
plt.title('Boxplot of Pollutant Distributions')
plt.savefig('pollutant_distributions.png')
plt.close()

# 10. Correlation Analysis ---
# Which variable is most correlated with PM2.5?
correlations = df[num_cols].corr()['PM2.5'].sort_values(ascending=False)
print("\n--- Correlation of Variables with PM2.5 ---")
print(correlations)

#Heatmap: Temperature vs. All Pollutants ---
plt.figure(figsize=(10, 8))

# Define the pollutants we want to compare with TEMP
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
temp_analysis_cols = ['TEMP'] + pollutants

# Calculate the correlation matrix specifically for these variables
temp_corr_matrix = df[temp_analysis_cols].corr()

# Create the heatmap
sns.heatmap(temp_corr_matrix[['TEMP']].sort_values(by='TEMP', ascending=False), 
            annot=True, 
            cmap='coolwarm', 
            fmt=".3f", 
            linewidths=0.5)

plt.title('Correlation: Temperature vs. All Pollutants', fontsize=14)
plt.savefig('temperature_all_pollutants_corr.png', dpi=300)
plt.close()

temp_corr = correlations['TEMP']


# 11. Exporting Results ---

# Export the combined and cleaned data to a single CSV
try:
    df.to_csv('PRSA_data.csv', index=False)
    print("\nSuccessfully exported 'PRSA_data.csv'")
except Exception as e:
    print(f"Error exporting CSV: {e}")


# 12. Export Analysis Results to Text File ---
analysis_filename = "PRSA_Analysis.txt"
with open(analysis_filename, "w") as f:
    f.write("==============================================\n")
    f.write("         PRSA DATA ANALYSIS REPORT           \n")
    f.write("==============================================\n\n")
    f.write("--- 1. DESCRIPTIVE STATISTICS ---\n")
    f.write(stats[['mean', 'median', 'std', 'min', 'max']].to_string())
    f.write("\n\n--- 2. AVERAGE POLLUTION LEVELS PER STATION ---\n")
    f.write(avg_pollution.to_string())
    f.write("\n\n--- 3. CORRELATION WITH PM2.5 ---\n")
    f.write(correlations.to_string())
    f.write(f"\n\n--- 4. CONCLUSIONS ---\n")
    f.write(f"Variable most correlated with PM2.5: {correlations.index[1]}\n")
    f.write(f"Does temperature affect pollution? Correlation: {temp_corr:.4f}\n")