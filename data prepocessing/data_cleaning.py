import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# 1. LOAD DATA
print("\nLOAD DATA")
df = pd.read_csv("data/sustainable_materials.csv")
print("Rows:", len(df))
print("Shape:", df.shape)
print(df.head())

# 2. MISSING VALUE SUMMARY
print("\nMISSING VALUE SUMMARY (BEFORE)")
missing_summary = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing %": (df.isnull().mean() * 100).round(2)
})
print(missing_summary)


# 3. HANDLE MISSING VALUES
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

# Numeric → median
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Categorical → mode
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Remove duplicates
df = df.drop_duplicates()

# Drop fully empty rows
df = df.dropna(how="all")

# Drop columns with >80% missing
cols_to_drop = df.columns[df.isnull().mean() > 0.80]
if len(cols_to_drop) > 0:
    df = df.drop(columns=cols_to_drop)
    print("Dropped columns (>80% missing):", list(cols_to_drop))

# Save STEP 1
df_step1 = df.copy()
df_step1.to_csv("step1_missing_values_handled.csv", index=False)
print("✔ Saved: step1_missing_values_handled.csv")


# 4. OUTLIER TREATMENT (IQR → MEDIAN)
print("\nOUTLIER TREATMENT")

for col in numeric_cols:
    if col not in df.columns:
        continue

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    mask = (df[col] < lower) | (df[col] > upper)
    if mask.any():
        df.loc[mask, col] = df[col].median()


# 5. NORMALIZATION 
print("\nNORMALIZATION")

numeric_cols_to_scale = numeric_cols.drop("material_id", errors="ignore")

scaler = MinMaxScaler()
df_step2 = df.copy()
df_step2[numeric_cols_to_scale] = scaler.fit_transform(
    df_step2[numeric_cols_to_scale]
)

df_step2.to_csv("step2_normalized_data.csv", index=False)
print("✔ Saved: step2_normalized_data.csv")

# 6. CATEGORICAL ENCODING
# (DO NOT encode material_id, type, subtype)
print("\nCATEGORICAL ENCODING")

exclude_encode = ["material_id", "material_type", "material_subtype"]

string_cols = df_step2.select_dtypes(include=["object"]).columns
encode_cols = [c for c in string_cols if c not in exclude_encode]

df_step3 = df_step2.copy()
encoders = {}

for col in encode_cols:
    le = LabelEncoder()
    df_step3[col] = le.fit_transform(df_step3[col])
    encoders[col] = le
    print(f"Encoded: {col}")

df_step3.to_csv("step3_encoded_data.csv", index=False)
print("✔ Saved: step3_encoded_data.csv")


# 7. CORRELATION MATRIX (NUMERIC ONLY)
print("\nCORRELATION MATRIX")

corr = df_step3.select_dtypes(include=["int64", "float64"]).corr()
print(corr)

corr.to_csv("correlation_matrix.csv")

print("\n✔ ALL STEPS COMPLETED SUCCESSFULLY")
