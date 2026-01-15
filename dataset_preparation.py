# ================================
# Milestone 2 (Week 3–4) & Module 4
# Enhanced ML Pipeline + MLflow
# EcoPackAI – Cost & CO2 Prediction (FIXED)
# ================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

import mlflow
import mlflow.sklearn
import joblib

# -------------------------------
# STEP 1: Load Dataset
# -------------------------------
df = pd.read_csv("fully_featured_materials.csv")

print("\nSTEP 1: Dataset Loaded")
print("Shape:", df.shape)
print(df.head())

# -------------------------------
# STEP 2: Feature Selection (FIXED)
# -------------------------------
# ONLY numeric physical & sustainability properties are ML features
numeric_features = [
    "strength_MPa",
    "weight_capacity_kg",
    "density_g_per_cm3",
    "biodegradability_score",
    "recyclability_percentage",
    "material_suitability_score"
]

X = df[numeric_features]
y_cost = df["cost_per_kg_usd"]
y_co2 = df["co2_emission_kgCO2_per_kg"]

print("\nSTEP 2: Features & Targets Selected")
print("Numeric Features:", numeric_features)
print("Categorical Features: NONE (by design)")

# -------------------------------
# STEP 3: Train–Test Split
# -------------------------------
X_train, X_test, y_cost_train, y_cost_test, y_co2_train, y_co2_test = train_test_split(
    X, y_cost, y_co2, test_size=0.2, random_state=42
)

print("\nSTEP 3: Train-Test Split Completed")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# -------------------------------
# STEP 4: Preprocessing Pipeline
# -------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features)
    ]
)

print("\nSTEP 4: Preprocessor created (numeric only)")

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
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("model", RandomForestRegressor(random_state=42))
    ])

    rf_param_grid = {
        "model__n_estimators": [1000],
        "model__max_depth": [None, 40],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt"]
    }

    rf_grid = GridSearchCV(
        rf_pipeline,
        rf_param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    rf_grid.fit(X_train, y_cost_train)

    best_rf_model = rf_grid.best_estimator_
    y_cost_pred = best_rf_model.predict(X_test)

    rmse_cost = np.sqrt(mean_squared_error(y_cost_test, y_cost_pred))
    mae_cost = mean_absolute_error(y_cost_test, y_cost_pred)
    r2_cost = r2_score(y_cost_test, y_cost_pred)

    cv_r2_cost = cross_val_score(best_rf_model, X, y_cost, cv=10, scoring="r2").mean()

    print("\nSTEP 6: Cost Prediction Results")
    print("Best Params:", rf_grid.best_params_)
    print(f"RMSE: {rmse_cost:.4f}")
    print(f"MAE : {mae_cost:.4f}")
    print(f"R²  : {r2_cost:.4f}")
    print(f"CV R²: {cv_r2_cost:.4f}")

    mlflow.log_params(rf_grid.best_params_)
    mlflow.log_metrics({
        "RMSE_cost": rmse_cost,
        "MAE_cost": mae_cost,
        "R2_cost": r2_cost,
        "CV_R2_cost": cv_r2_cost
    })

    mlflow.sklearn.log_model(best_rf_model, "rf_cost_model")

# -------------------------------
# STEP 7: CO₂ Prediction (XGBoost)
# -------------------------------
with mlflow.start_run(run_name="XGBoost_CO2_Prediction"):

    xgb_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("model", xgb.XGBRegressor(
            random_state=42,
            n_estimators=500,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1
        ))
    ])

    xgb_pipeline.fit(X_train, y_co2_train)

    y_co2_pred = xgb_pipeline.predict(X_test)

    rmse_co2 = np.sqrt(mean_squared_error(y_co2_test, y_co2_pred))
    mae_co2 = mean_absolute_error(y_co2_test, y_co2_pred)
    r2_co2 = r2_score(y_co2_test, y_co2_pred)

    cv_r2_co2 = cross_val_score(xgb_pipeline, X, y_co2, cv=10, scoring="r2").mean()

    print("\nSTEP 7: CO₂ Prediction Results")
    print(f"RMSE: {rmse_co2:.4f}")
    print(f"MAE : {mae_co2:.4f}")
    print(f"R²  : {r2_co2:.4f}")
    print(f"CV R²: {cv_r2_co2:.4f}")

    mlflow.log_metrics({
        "RMSE_co2": rmse_co2,
        "MAE_co2": mae_co2,
        "R2_co2": r2_co2,
        "CV_R2_co2": cv_r2_co2
    })

    mlflow.sklearn.log_model(xgb_pipeline, "xgb_co2_model")

# -------------------------------
# STEP 8: Material Ranking CSV
# -------------------------------
df_rank = df.loc[X_test.index, ["material_type", "material_subtype"]].copy()
df_rank[numeric_features] = X_test[numeric_features]

df_rank["predicted_cost"] = best_rf_model.predict(X_test)
df_rank["predicted_co2"] = xgb_pipeline.predict(X_test)

df_rank["material_rank_score"] = (
    0.5 * df_rank["predicted_cost"] +
    0.5 * df_rank["predicted_co2"]
)

df_rank = df_rank.sort_values("material_rank_score")
df_rank.to_csv("material_ranking.csv", index=False)

print("\nSTEP 8: Material Ranking Generated")
print(df_rank.head())

# -------------------------------
# STEP 9: Save Models
# -------------------------------
joblib.dump(best_rf_model, "rf_cost_model.pkl")
joblib.dump(xgb_pipeline, "xgb_co2_model.pkl")

print("\nPIPELINE EXECUTION COMPLETED SUCCESSFULLY")
