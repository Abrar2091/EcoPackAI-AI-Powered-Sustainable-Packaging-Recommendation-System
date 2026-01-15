# FULL CSV + POSTGRES VALIDATION SCRIPT (40+ CHECKS)

import pandas as pd
import psycopg2
from psycopg2 import sql

# CONFIG 
CSV_FILE = "C:/Users/Lenovo/OneDrive/Desktop/Project2/data/sustainable_materials.csv"
TABLE_NAME = "sustainable_materials"
EXPECTED_COLUMNS = [
    "material_id","material_type","material_subtype","strength_MPa","weight_capacity_kg",
    "density_g_per_cm3","cost_per_kg_usd","biodegradability_score",
    "co2_emission_kgCO2_per_kg","recyclability_percentage"
]
NUMERIC_COLS = [
    "strength_MPa","weight_capacity_kg","density_g_per_cm3","cost_per_kg_usd",
    "biodegradability_score","co2_emission_kgCO2_per_kg","recyclability_percentage"
]
PK_COL = "material_id"

FK_CONFIG = {
    "category_id": {
        "ref_table": "categories",
        "ref_column": "category_id"
    }
}

# CONNECT TO POSTGRES 
conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="pass",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

df = pd.read_csv(CSV_FILE)


# VALIDATION FUNCTIONS

def table_exists(table):
    cur.execute("SELECT to_regclass(%s);", (table,))
    return cur.fetchone()[0] is not None


def db_columns():
    cur.execute(sql.SQL("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = %s
        ORDER BY ordinal_position
    """), [TABLE_NAME])
    return cur.fetchall()


def validate_column_structure():
    db_cols = [c[0] for c in db_columns()]

    print("\n--- COLUMN VALIDATION ---")
    print("Table exists:", table_exists(TABLE_NAME))
    print("Column existence:", set(EXPECTED_COLUMNS).issubset(df.columns))
    print("Missing columns:", set(EXPECTED_COLUMNS)-set(df.columns))
    print("Extra CSV columns:", set(df.columns)-set(EXPECTED_COLUMNS))
    print("Column count match:", len(EXPECTED_COLUMNS)==len(df.columns))
    print("Column order correct:", EXPECTED_COLUMNS==list(df.columns))
    print("Column order mismatch warning:")
    if EXPECTED_COLUMNS!=list(df.columns):
        print(" → ORDER MISMATCH DETECTED!")


def validate_data_types():
    print("\n--- STRICT DATA TYPE VALIDATION ---")
    for col in NUMERIC_COLS:
        try: df[col]=df[col].astype(float)
        except: print(f"Invalid numeric type → {col}")


def missing_value_check():
    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())


def duplicate_check():
    print("\n--- DUPLICATE CHECK ---")
    print("Duplicate rows:", df.duplicated().sum())
    print("Duplicate PK:", df[PK_COL].duplicated().sum())


def string_format_check():
    print("\n--- STRING FORMAT VALIDATION ---")
    has_issue = False

    for col in ["material_type", "material_subtype"]:
        bad = df[df[col].str.contains(r"[^a-zA-Z0-9 \-]", regex=True, na=False)]
        if not bad.empty:
            has_issue = True
            print(f"Invalid characters found in column: {col}")
            print(bad[[col]])

    if not has_issue:
        print("✔ All string columns passed formatting validation.")


def value_range_check():
    print("\n--- VALUE RANGE VALIDATION ---")
    issues = False

    if not df['recyclability_percentage'].between(0,100).all():
        issues = True
        print("❌ Invalid recyclability percentage found!")

    if not df['biodegradability_score'].between(0,100).all():
        issues = True
        print("❌ Invalid biodegradability score found!")

    if not issues:
        print("✔ All numeric ranges are valid.")

def outlier_check():
    print("\n--- OUTLIER DETECTION ---")
    issues = False

    for col in NUMERIC_COLS:
        mean, std = df[col].mean(), df[col].std()
        outliers = df[(df[col] > mean + 3*std) | (df[col] < mean - 3*std)]

        if not outliers.empty:
            issues = True
            print(f"❌ Outliers detected in {col}:")
            print(outliers[[col]])

    if not issues:
        print("✔ No outliers detected.")



def fk_validation():
    print("\n--- FOREIGN KEY VALIDATION ---")
    for fk, ref in FK_CONFIG.items():
        if fk not in df.columns:
            print(f"FK column missing: {fk}")
            continue
        cur.execute(sql.SQL(f"SELECT {ref['ref_column']} FROM {ref['ref_table']}"))
        valid = set(r[0] for r in cur.fetchall())
        invalid = set(df[fk]) - valid
        if invalid:
            print(f"Invalid FK values in {fk}: {invalid}")


def db_constraint_report():
    print("\n--- DB CONSTRAINT REPORT ---")
    cur.execute(sql.SQL("""
        SELECT conname, contype
        FROM pg_constraint
        WHERE conrelid=%s::regclass
    """), [TABLE_NAME])
    print(cur.fetchall())


def not_null_validation():
    print("\n--- NOT NULL VALIDATION ---")
    issues = False

    for col in EXPECTED_COLUMNS:
        if df[col].isnull().any():
            issues = True
            null_count = df[col].isnull().sum()
            print(f"❌ Null values found in column '{col}': {null_count}")

    if not issues:
        print("✔ No NULL values found in required columns.")



def row_count_check():
    print("\n--- ROW COUNT / EMPTY CHECK ---")
    print("Row count:", len(df))
    if len(df)==0:
        print("WARNING: CSV is empty!")


# RUN ALL VALIDATIONS

validate_column_structure()
validate_data_types()
missing_value_check()
duplicate_check()
string_format_check()
value_range_check()
outlier_check()
fk_validation()
db_constraint_report()
not_null_validation()
row_count_check()

print("\n✔ ALL VALIDATIONS COMPLETED ✔")
