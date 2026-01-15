# VALIDATION SCRIPT FOR DATASET 2 (product_categories_dataset.csv)

import pandas as pd
import psycopg2
from psycopg2 import sql

######################## CONFIG ########################
CSV_FILE = "C:/Users/Lenovo/OneDrive/Desktop/Project2/data/product_categories_dataset.csv"
TABLE_NAME = "product_categories"

EXPECTED_COLUMNS = [
    "category_id",
    "category_name",
    "common_products",
    "packaging_requirements",
    "fragility_level",
    "ideal_material_types"
]

NUMERIC_COLS = ["category_id", "fragility_level"]
PK_COL = "category_id"

############### CONNECT TO POSTGRES ####################
conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="pass",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

df = pd.read_csv(CSV_FILE)

########################################################
# VALIDATION FUNCTIONS
########################################################

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
    print("\n--- COLUMN VALIDATION ---")

    print("Table exists:", table_exists(TABLE_NAME))

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(EXPECTED_COLUMNS)

    print("Missing columns:", missing)
    print("Extra CSV columns:", extra)
    print("Column existence:", len(missing) == 0)
    print("Column count match:", len(EXPECTED_COLUMNS) == len(df.columns))

    print("Column order correct:", EXPECTED_COLUMNS == list(df.columns))
    if EXPECTED_COLUMNS != list(df.columns):
        print(" → ORDER MISMATCH DETECTED!")


def validate_data_types():
    print("\n--- STRICT DATA TYPE VALIDATION ---")

    for col in NUMERIC_COLS:
        if col not in df.columns:
            print(f"⚠ Column missing → {col} (skipping type check)")
            continue
        try:
            df[col] = df[col].astype(float)
        except:
            print(f"❌ Invalid numeric type → {col}")


def missing_value_check():
    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())


def duplicate_check():
    print("\n--- DUPLICATE CHECK ---")
    print("Duplicate rows:", df.duplicated().sum())

    if PK_COL not in df.columns:
        print(f"⚠ PK column '{PK_COL}' not found → skipping PK duplicate check")
        return

    print("Duplicate PK:", df[PK_COL].duplicated().sum())


def string_format_check():
    print("\n--- STRING FORMAT VALIDATION ---")
    issue = False

    string_cols = [
        "category_name", "common_products",
        "packaging_requirements", "ideal_material_types"
    ]

    for col in string_cols:
        if col not in df.columns:
            print(f"⚠ Missing column (skipped): {col}")
            continue

        bad = df[df[col].astype(str).str.contains(r"[^a-zA-Z0-9 ,;:\-\&]", regex=True)]

        if not bad.empty:
            issue = True
            print(f"❌ Invalid characters found in column: {col}")
            print(bad[[col]])

    if not issue:
        print("✔ All string columns passed formatting validation.")


def value_range_check():
    print("\n--- VALUE RANGE VALIDATION ---")

    if "fragility_level" not in df.columns:
        print("⚠ Missing fragility_level, skipping range validation")
        return

    if not df["fragility_level"].between(1, 5).all():
        print("❌ fragility_level must be between 1 and 5")
    else:
        print("✔ All numeric ranges are valid.")


def outlier_check():
    print("\n--- OUTLIER DETECTION ---")
    issue = False

    for col in NUMERIC_COLS:
        if col not in df.columns:
            print(f"⚠ Missing column (skipping): {col}")
            continue

        mean, std = df[col].mean(), df[col].std()
        outliers = df[(df[col] > mean + 3*std) | (df[col] < mean - 3*std)]

        if not outliers.empty:
            issue = True
            print(f"❌ Outliers detected in {col}:")
            print(outliers[[col]])

    if not issue:
        print("✔ No outliers detected.")


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
    issue = False

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            print(f"⚠ Missing column (skip): {col}")
            continue

        if df[col].isnull().any():
            issue = True
            print(f"❌ Null values found in column: {col}")

    if not issue:
        print("✔ No NULL values found.")


def row_count_check():
    print("\n--- ROW COUNT / EMPTY CHECK ---")
    print("Row count:", len(df))
    if len(df) == 0:
        print("❌ CSV is empty!")


########################################################
# RUN ALL VALIDATIONS
########################################################

validate_column_structure()
validate_data_types()
missing_value_check()
duplicate_check()
string_format_check()
value_range_check()
outlier_check()
db_constraint_report()
not_null_validation()
row_count_check()

print("\n✔ ALL VALIDATIONS COMPLETED ✔")
