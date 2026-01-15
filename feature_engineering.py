# FULL DATA CLEANING + FEATURE ENGINEERING

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. LOAD DATA (already cleaned + encoded)
df = pd.read_csv("step3_encoded_data.csv")
print("Loaded rows:", len(df))
print("Columns available:", df.columns.tolist())

# Create scaler
scaler = MinMaxScaler()

# 2. MISSING VALUE CHECK
print("\nMissing values check:")
print(df.isnull().sum())

# 3. REMOVE DUPLICATES
duplicates = df.duplicated().sum()
print("\nDuplicate rows found:", duplicates)

df = df.drop_duplicates()
print("After removal, rows:", len(df))

# 4. FEATURE ENGINEERING
df_cleaned = df.copy()

# 4a. CO₂ Impact Index (lower CO₂ → higher score)
df_cleaned["co2_impact_index"] = 1 - scaler.fit_transform(
    df_cleaned[["co2_emission_kgCO2_per_kg"]]
)
df_cleaned["co2_impact_index"] = df_cleaned["co2_impact_index"].round(2)


# 4b. Cost Efficiency Index (lower cost → higher score)
df_cleaned["cost_efficiency_index"] = 1 - scaler.fit_transform(
    df_cleaned[["cost_per_kg_usd"]]
)
df_cleaned["cost_efficiency_index"] = df_cleaned["cost_efficiency_index"].round(2)

# 4c. Normalize features for suitability score
df_cleaned["strength_norm"] = scaler.fit_transform(df_cleaned[["strength_MPa"]])
df_cleaned["weight_capacity_norm"] = scaler.fit_transform(df_cleaned[["weight_capacity_kg"]])
df_cleaned["density_norm"] = scaler.fit_transform(df_cleaned[["density_g_per_cm3"]])
df_cleaned["biodegradability_norm"] = scaler.fit_transform(df_cleaned[["biodegradability_score"]])
df_cleaned["recyclability_norm"] = scaler.fit_transform(df_cleaned[["recyclability_percentage"]])


# 4d. Material Suitability Score (Weighted)
df_cleaned["material_suitability_score"] = (
    0.25 * df_cleaned["co2_impact_index"] +
    0.25 * df_cleaned["cost_efficiency_index"] +
    0.15 * df_cleaned["strength_norm"] +
    0.10 * df_cleaned["weight_capacity_norm"] +
    0.10 * df_cleaned["density_norm"] +
    0.10 * df_cleaned["biodegradability_norm"] +
    0.05 * df_cleaned["recyclability_norm"]
)

df_cleaned["material_suitability_score"] = (
    df_cleaned["material_suitability_score"]
    .clip(0, 1)
    .round(2)
)

# 5. EXPORT FINAL DATASET
df_cleaned.to_csv("fully_featured_materials.csv", index=False)
print("\n✔ Fully featured dataset saved as: fully_featured_materials.csv")


# 6. SAMPLE OUTPUT
print("\nSample of computed indices:")
print(
    df_cleaned[
        [
            "material_id",
            "material_type",
            "co2_impact_index",
            "cost_efficiency_index",
            "material_suitability_score"
        ]
    ].head()
)
