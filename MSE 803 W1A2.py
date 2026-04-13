import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder

# 1. Load the dataset
try:
    df = pd.read_csv('Housing.csv')
    print("Dataset successfully loaded!\n")
except FileNotFoundError:
    print("Error: The file 'Housing.csv' was not found.")
    exit() # Interrompe se não achar o arquivo

# 2. Preprocessing: Convert Categorical (Yes/No) to Binary (1/0)
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# --- CRITICAL STEP: Define X and y BEFORE using them ---
# y is our target (Price)
y = df['price']
# X is our features (Everything except price)
X = df.drop('price', axis=1)

# Now we can add the constant to X
X = sm.add_constant(X)

# 3. Initial Data Inspection & Statistics
print("--- First 5 Rows of the Dataset ---")
print(df.head())

print("\n--- Statistical Summary ---")
print(df.describe())

# 4. Correlation Analysis
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# 5. Price Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['price'], kde=True, color='blue')
plt.title("Distribution of Housing Prices")
plt.show()

# 6. Fit the Multiple Linear Regression Model (OLS)
# Note: Now y and X are properly defined
model = sm.OLS(y, X).fit()
print("\n--- Regression Results ---")
print(model.summary())

# 7. Identifying the most influential variable (T-values)
print("\n--- Impact Analysis (T-values) ---")
# High T-value = High influence on price
print(model.tvalues.abs().sort_values(ascending=False))