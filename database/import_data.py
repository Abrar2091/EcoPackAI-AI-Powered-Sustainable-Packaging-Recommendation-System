"""
Validates CSV schema and loads data into PostgreSQL using COPY command.
Integrates CSV/Excel material data and validates schema.
"""

import os
import pandas as pd
import psycopg2

# FIXED: correct import
from config import get_connection

# CSV FILES (DEFINE FIRST!)
materials_file = "./Data/sustainable_materials.csv"
autoliv_file = "./Data/product_categories_dataset.csv"

# DEBUG — Print exact paths
print("\n=== DEBUG PATH CHECK ===")
print("Materials CSV expected at:", os.path.abspath(materials_file))
print("Does materials.csv exist? ", os.path.exists(materials_file))
print("Autoliv CSV expected at :", os.path.abspath(autoliv_file))
print("Does autoliv CSV exist? ", os.path.exists(autoliv_file))
print("=========================\n")

# STOP if files missing
if not os.path.exists(materials_file):
    raise FileNotFoundError(f"❌ materials.csv NOT FOUND at {materials_file}")

if not os.path.exists(autoliv_file):
    raise FileNotFoundError(f"❌ autoliv_materials.csv NOT FOUND at {autoliv_file}")

# Expected schema for materials.csv (NO SERIAL ID)
materials_schema = [
    "material_type","material_subtype","strength_MPa",
    "weight_capacity_kg","density_g_per_cm3","cost_per_kg_usd",
    "biodegradability_score","co2_emission_kgCO2_per_kg",
    "recyclability_percentage"
]

# Expected schema for autoliv file
autoliv_schema = [
    "category_name","common_products",
    "packaging_requirements","fragility_level","ideal_material_types"
]

def validate_csv(path, expected_cols):
    df = pd.read_csv(path)

    # Remove ID if exists in CSV
    df_columns = df.columns.tolist()
    if df_columns[0].lower() in ("id", "material_id", "category_id"):
        df_columns = df_columns[1:]

    if df_columns != expected_cols:
        raise ValueError(
            f"\n❌ SCHEMA MISMATCH IN {path}\n"
            f"Expected: {expected_cols}\n"
            f"Got:      {df_columns}\n"
        )
    
    print(f"✓ Schema validated: {path}")
    return df

def load_csv_to_db(csv_path, table_name):
    conn = get_connection()
    cur = conn.cursor()

    print(f"\nInserting {csv_path} into {table_name} ...")

    with open(csv_path, "r", encoding="utf-8") as f:
        next(f)  
        cur.copy_expert(f"COPY {table_name} FROM STDIN WITH CSV HEADER", f)

    conn.commit()
    cur.close()
    conn.close()

    print(f"✓ Loaded {csv_path} into {table_name}")

if __name__ == "__main__":

    validate_csv(materials_file, materials_schema)
    validate_csv(autoliv_file, autoliv_schema)

    load_csv_to_db(materials_file, "sustainability.materials")
    load_csv_to_db(autoliv_file, "sustainability.autoliv_materials")

