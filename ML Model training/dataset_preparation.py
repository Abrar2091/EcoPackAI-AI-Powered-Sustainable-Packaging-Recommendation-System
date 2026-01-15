# ================================
# Milestone 2 (Week 3–4) & Module 4
# Fully Corrected ML Pipeline + MLflow
# EcoPackAI – Cost & CO2 Prediction
# ================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb
import mlflow
import mlflow.sklearn

# -------------------------------
# STEP 1: Load Dataset
# -------------------------------
df = pd.read_csv("fully_featured_materials.csv")

print("\nSTEP 1: Dataset Loaded")
print("Shape:", df.shape)
print(df.head())

# -------------------------------
# STEP 2: Feature Selection
# -------------------------------
feature_cols = [
    "strength_MPa",
    "weight_capacity_kg",
    "density_g_per_cm3",
    "biodegradability_score",
    "recyclability_percentage",
    "material_suitability_score"
]

X = df[feature_cols]
y_cost = df["cost_per_kg_usd"]
y_co2 = df["co2_emission_kgCO2_per_kg"]

print("\nSTEP 2: Features & Targets Selected")
print("Features:", feature_cols)

# -------------------------------
# STEP 3: Train–Test Split
# -------------------------------
X_train, X_test, y_cost_train, y_cost_test, y_co2_train, y_co2_test = train_test_split(
    X, y_cost, y_co2, test_size=0.2, random_state=42
)

print("\nSTEP 3: Train-Test Split Completed")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Save split datasets
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_cost_train.to_csv("y_cost_train.csv", index=False)
y_cost_test.to_csv("y_cost_test.csv", index=False)
y_co2_train.to_csv("y_co2_train.csv", index=False)
y_co2_test.to_csv("y_co2_test.csv", index=False)

print("Train-test datasets saved as CSV files")

# -------------------------------
# STEP 4: Data Pipeline & Scaling
# -------------------------------
preprocessor = ColumnTransformer(
    transformers=[("num", StandardScaler(), feature_cols)]
)

print("\nSTEP 4: Data Pipeline Created with StandardScaler")

# -------------------------------
# STEP 5: MLflow Experiment Setup
# -------------------------------
mlflow.set_experiment("EcoPackAI_Material_Prediction")
print("\nSTEP 5: MLflow Experiment Initialized")

# -------------------------------
# STEP 6: Cost Prediction (Random Forest)
# -------------------------------
with mlflow.start_run(run_name="RandomForest_Cost_Prediction"):

    rf_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=42
        ))
    ])

    rf_pipeline.fit(X_train, y_cost_train)
    y_cost_pred = rf_pipeline.predict(X_test)

    mse_cost = mean_squared_error(y_cost_test, y_cost_pred)
    rmse_cost = np.sqrt(mse_cost)
    mae_cost = mean_absolute_error(y_cost_test, y_cost_pred)
    r2_cost = r2_score(y_cost_test, y_cost_pred)

    print("\nSTEP 6: Cost Prediction Results (Random Forest)")
    print(f"RMSE: {rmse_cost:.4f}")
    print(f"MAE : {mae_cost:.4f}")
    print(f"R²  : {r2_cost:.4f}")

    mlflow.log_metrics({
        "RMSE": rmse_cost,
        "MAE": mae_cost,
        "R2": r2_cost
    })

    mlflow.sklearn.log_model(
    sk_model=rf_pipeline,
    name="rf_cost_model"
    )

# -------------------------------
# STEP 7: CO₂ Prediction (XGBoost)
# -------------------------------
with mlflow.start_run(run_name="XGBoost_CO2_Prediction"):

    xgb_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ))
    ])

    xgb_pipeline.fit(X_train, y_co2_train)
    y_co2_pred = xgb_pipeline.predict(X_test)

    mse_co2 = mean_squared_error(y_co2_test, y_co2_pred)
    rmse_co2 = np.sqrt(mse_co2)
    mae_co2 = mean_absolute_error(y_co2_test, y_co2_pred)
    r2_co2 = r2_score(y_co2_test, y_co2_pred)

    print("\nSTEP 7: CO₂ Prediction Results (XGBoost)")
    print(f"RMSE: {rmse_co2:.4f}")
    print(f"MAE : {mae_co2:.4f}")
    print(f"R²  : {r2_co2:.4f}")

    mlflow.log_metrics({
        "RMSE": rmse_co2,
        "MAE": mae_co2,
        "R2": r2_co2
    })

    
    mlflow.sklearn.log_model(
        sk_model=xgb_pipeline,
        name="xgb_co2_model"
    )

# -------------------------------
# STEP 8: Material Ranking System
# -------------------------------
df_rank = X_test.copy()

df_rank["predicted_cost"] = rf_pipeline.predict(X_test)
df_rank["predicted_co2"] = xgb_pipeline.predict(X_test)

# Lower score = better material

df_rank["material_rank_score"] = (
    0.5 * df_rank["predicted_cost"] +
    0.5 * df_rank["predicted_co2"]
)

df_rank = df_rank.sort_values("material_rank_score")

# Save ranking results
df_rank.to_csv("material_ranking.csv", index=False)

print("\nSTEP 8: Material Ranking Generated")
print(df_rank.head())

print("\nPIPELINE EXECUTION COMPLETED SUCCESSFULLY")

# -------------------------------
# END OF SCRIPT
# -------------------------------