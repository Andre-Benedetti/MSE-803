import pandas as pd

# 1. Load the original dataset verbatim
df = pd.read_csv('Wellbeing_data_all.csv')
df_clean = df.dropna(subset=['Estimate(percent)'])

# 2. Filter strictly for the target wellbeing measure
df_sat = df_clean[df_clean['Wellbeing measure'] == 'Overall life satisfaction']

# 3. Filter out 'Mean rating'
df_filtered = df_sat[df_sat['Wellbeing measure category'] != 'Mean rating']

# 4. Enforce strict ascending categorical order starting from '0 to 6' up to '10'
category_order = ['0 to 6', '7', '8', '9', '10']
df_filtered['Wellbeing measure category'] = pd.Categorical(
    df_filtered['Wellbeing measure category'], 
    categories=category_order, 
    ordered=True
)

# =====================================================================
# MASTER SUMMARY TABLE: ORDERED DISTRIBUTION OF PROPORTIONS
# =====================================================================
print("=====================================================================")
print("--- MASTER SUMMARY TABLE: OVERALL LIFE SATISFACTION (ASCENDING) ---")
print("=====================================================================")
print("Data sorted sequentially starting from '0 to 6' through to '10'.\n")

# Pivot and explicitly sort by index level to apply categorical ordering
master_pivot = df_filtered.pivot(
    index=['Demographic', 'Demographic category', 'Wellbeing measure category'], 
    columns='Year', 
    values='Estimate(percent)'
).sort_index(level=['Demographic', 'Demographic category', 'Wellbeing measure category'])

# Display the entire ordered master overview table
print(master_pivot.to_string())
print("\n" + "="*70 + "\n\n")


# =====================================================================
# INDIVIDUAL BREAKDOWNS (Detailed Views with Ascending Brackets)
# =====================================================================
print("--- INDIVIDUAL BREAKDOWNS (ASCENDING SCORE BRACKETS) ---")
demographics = df_filtered['Demographic'].unique()

for demo in demographics:
    df_demo = df_filtered[df_filtered['Demographic'] == demo]
    
    # Pivot and sort explicitly by the index categories
    pivot_table = df_demo.pivot(
        index=['Demographic category', 'Wellbeing measure category'], 
        columns='Year', 
        values='Estimate(percent)'
    ).sort_index(level=['Demographic category', 'Wellbeing measure category'])
    
    print(f"### Demographic Dimension Detailed View: {demo}")
    print("-" * 75)
    print(pivot_table)
    print("\n" + "="*85 + "\n")

# =====================================================================
#  MULTIDIMENSIONAL INFLUENCE ANALYSIS
# =====================================================================
print("=====================================================================")
print("--- MULTIDIMENSIONAL INFLUENCE ANALYSIS ---")
print("=====================================================================")
print("Ranking demographics by how much they shift the satisfaction brackets.")
print("Higher variance/gaps = Stronger influence on the Wellbeing category.\n")

influence_metrics = []

for demo in demographics:
    df_demo = df_filtered[df_filtered['Demographic'] == demo]
    
    # Calculate the variation (Standard Deviation) between sub-categories for each bracket and year
    std_by_bracket_year = df_demo.groupby(['Wellbeing measure category', 'Year'])['Estimate(percent)'].std()
    avg_std = std_by_bracket_year.mean()
    
    # Find the maximum percentage point gap (Max - Min) inside any bracket for this demographic
    max_val = df_demo.groupby(['Wellbeing measure category', 'Year'])['Estimate(percent)'].max()
    min_val = df_demo.groupby(['Wellbeing measure category', 'Year'])['Estimate(percent)'].min()
    max_gap = (max_val - min_val).max()
    
    influence_metrics.append({
        'Demographic Dimension': demo,
        'Avg Category Variance (Std Dev)': avg_std,
        'Max Internal Gap (Percentage Points)': max_gap
    })

# Convert to DataFrame, drop total population baseline control, and sort by influence
df_influence = pd.DataFrame(influence_metrics)
df_influence = df_influence[df_influence['Max Internal Gap (Percentage Points)'] > 0]
df_influence = df_influence.sort_values(by='Max Internal Gap (Percentage Points)', ascending=False)

print(df_influence.to_string(index=False))
print("\n" + "="*70 + "\n")

# =====================================================================
#  MULTIVARIATE MULTIPLE REGRESSION ANALYSIS
# =====================================================================
import statsmodels.api as sm
import statsmodels.formula.api as smf

print("=====================================================================")
print("--- MULTIVARIATE MULTIPLE REGRESSION ANALYSIS ---")
print("=====================================================================")
print("Modeling multiple dependent satisfaction brackets simultaneously.")
print("Reference category dropped to avoid perfect collinearity: Bracket '7'\n")

# Recreating the source data directly to prevent empty variable inheritance
df_sat_raw = df[df['Wellbeing measure'] == 'Overall life satisfaction']
df_reg_source = df_sat_raw[df_sat_raw['Wellbeing measure category'] != 'Mean rating'].dropna(subset=['Estimate(percent)'])

# 1. Prepare data for the multivariate regression framework
df_reg_prep = df_reg_source.pivot(
    index=['Demographic', 'Demographic category', 'Year'],
    columns='Wellbeing measure category',
    values='Estimate(percent)'
).reset_index()

# Clean up column names to avoid syntax errors in formulas (e.g., '0 to 6' -> 'bracket_0_to_6')
df_reg_prep.columns.name = None
df_reg_prep = df_reg_prep.rename(columns={
    '0 to 6': 'bracket_0_to_6',
    '7': 'bracket_7',
    '8': 'bracket_8',
    '9': 'bracket_9',
    '10': 'bracket_10'
})

# Standardize column types for string manipulation in statsmodels formulas
df_reg_prep['Demographic'] = df_reg_prep['Demographic'].astype(str)
df_reg_prep['Year'] = df_reg_prep['Year'].astype(int)

# Filter out baseline control row
df_reg_analysis = df_reg_prep[df_reg_prep['Demographic'] != 'Total population'].copy()

# Fill individual missing metric cells with 0 to maintain regression integrity
dependent_brackets = ['bracket_0_to_6', 'bracket_8', 'bracket_9', 'bracket_10']
df_reg_analysis[dependent_brackets] = df_reg_analysis[dependent_brackets].fillna(0)

# 3. Fit a separate Ordinary Least Squares (OLS) regression for each target equation
# Using Demographic and Year as independent variables (X)
regression_results = {}

for bracket in dependent_brackets:
    # Formula uses C() to automatically convert categorical dimensions into dummy variables
    formula = f"{bracket} ~ C(Demographic) + Year"
    model = smf.ols(formula, data=df_reg_analysis).fit()
    regression_results[bracket] = model

# 4. Display a streamlined summary map showing coefficients and significance (p-values)
print(f"{'Dependent Bracket':<16} | {'Predictor / Feature':<46} | {'Coef':<8} | {'p-value':<7}")
print("-" * 88)

for bracket, model in regression_results.items():
    # Extract coefficients and p-values from the fitted statsmodels wrapper
    params = model.params
    pvalues = model.pvalues
    
    for feature in params.index:
        # Omit print clutter from the baseline intercept constant
        if feature == 'Intercept':
            continue
            
        coef_val = params[feature]
        p_val = pvalues[feature]
        
        # Clean up the output feature name labels for enhanced scannability
        clean_feature = feature.replace("C(Demographic)[T.", "").replace("]", "")
        
        # Highlight significant parameters running with a alpha threshold under 5%
        sig_marker = "*" if p_val < 0.05 else " "
        
        print(f"{bracket:<16} | {clean_feature:<46} | {coef_val:>7.3f} | {p_val:>6.4f} {sig_marker}")
    print("-" * 88)

print("\n* Standard significance flag applied for features maintaining a p-value less than 0.05.")
print("=====================================================================\n")