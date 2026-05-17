# ==========================================
# STEP 1: Import (Import libraries)
# ==========================================
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC

# ==========================================
# STEP 2: Load, Clean, and Export Data
# ==========================================
iris_built_in = datasets.load_iris()

# Convert the built-in data into a Pandas DataFrame for cleaning
df = pd.DataFrame(iris_built_in.data, columns=iris_built_in.feature_names)

# Assign the numeric targets first so it can be mapped and exported properly
df["species_code"] = iris_built_in.target

# Map the target numbers (0, 1, 2) to actual text species names
species_names = [iris_built_in.target_names[i] for i in iris_built_in.target]
df["species"] = species_names

print("--- Data Cleaning & Exploration Process ---")
print(f"Original dataset shape: {df.shape}\n")

# --- Export the Species Mapping Reference File ---
print("Generating species mapping reference file...")
# Select only the name and numeric code columns, then keep unique rows
mapping_df = df[["species", "species_code"]].drop_duplicates().reset_index(drop=True)
# Rename columns explicitly for the English requirements
mapping_df.columns = ["Species Name", "Assigned Numeric Value"]

# Get current script directory to save files dynamically in the same folder
script_dir = os.path.dirname(os.path.abspath(__file__))
mapping_file_path = os.path.join(script_dir, "species_mapping.csv")

# Export mapping reference to CSV
mapping_df.to_csv(mapping_file_path, index=False)
print(f"Success! Mapping file saved to:\n-> {mapping_file_path}\n")

# --- Dataset Structure Information Logging ---
print("Dataset Structure Details:")
print(f"Total number of columns: {len(df.columns)}")
print("-" * 65)
# Loop through columns to display name, count of non-null values, and data type
for col in df.columns:
    col_name = col
    value_count = df[col].count()  # Counts non-null values
    data_type = df[col].dtype
    print(
        f"Column: {col_name:<25} | Values: {value_count:<5} | Data Type: {data_type}"
    )
print("-" * 65 + "\n")

# 1. Check and drop rows with any missing values (NaNs)
if df.isnull().values.any():
    print(f"Found {df.isnull().sum().sum()} missing values. Dropping those rows...")
    df.dropna(inplace=True)
else:
    print("No missing values found.")

# 2. Clean up text data (removes any accidental leading/trailing spaces)
df["species"] = df["species"].str.strip()

# 3. Remove duplicate rows to avoid overfitting
# Using a specific subset to avoid counting the temporary columns incorrectly
features_and_target = list(iris_built_in.feature_names) + ["species"]
duplicates_count = df.duplicated(subset=features_and_target).sum()
if duplicates_count > 0:
    print(f"Found {duplicates_count} duplicate rows. Removing them...")
    df.drop_duplicates(subset=features_and_target, inplace=True)

print(f"Cleaned dataset shape: {df.shape}\n")

# Separate features (X) and target labels (y) from the cleaned DataFrame
X = df[iris_built_in.feature_names]
y = df["species"]

# --- Export the Complete Cleaned Dataset with Names and Numerical Codes ---
print("Exporting complete cleaned dataset with names and numeric codes...")
exported_df = df.copy()

# Reorder and rename columns for the final clean report delivery
columns_order = list(iris_built_in.feature_names) + ["species", "species_code"]
exported_df = exported_df[columns_order]
exported_df.columns = list(iris_built_in.feature_names) + [
    "Species Name",
    "Assigned Numeric Value",
]

cleaned_dataset_path = os.path.join(
    script_dir, "iris_cleaned_dataset_with_codes.csv"
)
exported_df.to_csv(cleaned_dataset_path, index=False)
print(f"Success! Cleaned full dataset saved to:\n-> {cleaned_dataset_path}\n")

# Splitting the dataset into 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# STEP 3: Train (Train SVM classifier)
# ==========================================
svm_model = SVC(kernel="linear")
svm_model.fit(X_train, y_train)
print("Model training completed successfully!\n")

# ==========================================
# STEP 4: Predict and Advanced Evaluation
# ==========================================
# Make predictions on the test set
predictions = svm_model.predict(X_test)

# 1. Accuracy Score
accuracy = accuracy_score(y_test, predictions)

# 2. Precision and Recall (using 'macro' average for multi-class classification)
precision = precision_score(y_test, predictions, average="macro")
recall = recall_score(y_test, predictions, average="macro")

# 3. Confusion Matrix (Raw Data)
conf_matrix = confusion_matrix(y_test, predictions)

# 4. Cross-Validation for Reliability (using 5 folds on the whole dataset)
cv_scores = cross_val_score(svm_model, X, y, cv=5)

# Printing the evaluation metrics to the terminal
print("-" * 50)
print("EVALUATION METRICS COMPLETED")
print("-" * 50)
print(f"Accuracy Score:          {accuracy * 100:.2f}%")
print(f"Precision (Macro Avg):   {precision * 100:.2f}%")
print(f"Recall (Macro Avg):      {recall * 100:.2f}%")
print(f"Cross-Validation Scores: {[round(score * 100, 2) for score in cv_scores]}")
print(f"Mean CV Reliability:     {cv_scores.mean() * 100:.2f}%")
print("-" * 50)

print("\nDetailed Classification Report:")
print(classification_report(y_test, predictions))

# ==========================================
# STEP 5: Create Plots and Save Figures
# ==========================================
print("\nGenerating evaluation graphics...")

# --- Plot 1: Feature Analysis (Sepals vs Petals) ---
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

sns.scatterplot(
    ax=axes[0],
    data=df,
    x="sepal length (cm)",
    y="sepal width (cm)",
    hue="species",
    style="species",
    palette="deep",
    s=90,
)
axes[0].set_title("Sepal Length vs Sepal Width", fontsize=13, fontweight="bold")
axes[0].get_legend().remove()

sns.scatterplot(
    ax=axes[1],
    data=df,
    x="petal length (cm)",
    y="petal width (cm)",
    hue="species",
    style="species",
    palette="deep",
    s=90,
)
axes[1].set_title("Petal Length vs Petal Width", fontsize=13, fontweight="bold")
axes[1].legend(title="Flower Species", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plot_1_path = os.path.join(script_dir, "iris_features_analysis.png")
plt.savefig(plot_1_path, dpi=300)
plt.close()

# --- Plot 2: Confusion Matrix Heatmap ---
plt.figure(figsize=(7, 5))
labels = sorted(df["species"].unique())

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    cbar=False,
    annot_kws={"size": 14, "weight": "bold"},
)

plt.title("Confusion Matrix Heatmap", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Predicted Label", fontsize=12, labelpad=10)
plt.ylabel("True Label", fontsize=12, labelpad=10)
plt.tight_layout()

plot_2_path = os.path.join(script_dir, "iris_confusion_matrix.png")
plt.savefig(plot_2_path, dpi=300)
plt.close()

print("Success! Both evaluation figures saved to directory:")
print(f"-> {plot_1_path}")
print(f"-> {plot_2_path}")