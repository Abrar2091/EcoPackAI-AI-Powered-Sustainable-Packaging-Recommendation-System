# 📦 EcoPackAI  
## Module 1: Data Collection & Management  
(Material Database + Product Attributes)

---

## 📌 Module Overview

Module 1 is the data foundation layer of EcoPackAI.  
It is responsible for collecting, validating, and storing eco-friendly packaging material data and product attributes into a structured PostgreSQL database.

---

## 🎯 Objectives

- Collect eco-friendly packaging material data (CSV / Excel)
- Store data in PostgreSQL
- Validate schema and data integrity
- Enable easy retrieval for downstream ML modules

---

## 🧱 Data Attributes

### Packaging Material Attributes

- material_id  
- material_name  
- material_type  
- strength_rating  
- weight_capacity_kg  
- biodegradability_score  
- recyclability_percent  
- co2_emission_score  
- cost_per_unit  

### Product Attributes

- product_id  
- category  
- fragility_level  
- weight_kg  
- shipping_distance_km  

---

## 🗂 Folder Structure

```
EcoPackAI/
└── module1_data_collection/
    ├── data/
    │   ├── materials.csv
    │   └── products.csv
    ├── db/
    │   ├── db_config.py
    │   └── create_tables.sql
    ├── ingest_data.py
    └── validate_data.py
```

---

## 🛠 Packages Used

| Package | Version |
|------|--------|
| python | 3.10 |
| pandas | 2.1.1 |
| psycopg2-binary | 2.9.9 |
| sqlalchemy | 2.0.23 |
| python-dotenv | 1.0.0 |

---

## ⚙️ Installation

```bash
pip install pandas==2.1.1 psycopg2-binary==2.9.9 sqlalchemy==2.0.23 python-dotenv==1.0.0
```

---

## 🗄 Database Schema

### materials table

```sql
CREATE TABLE materials (
    material_id SERIAL PRIMARY KEY,
    material_name VARCHAR(100),
    material_type VARCHAR(50),
    strength_rating INT,
    weight_capacity_kg FLOAT,
    biodegradability_score FLOAT,
    recyclability_percent FLOAT,
    co2_emission_score FLOAT,
    cost_per_unit FLOAT
);
```

### products table

```sql
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    category VARCHAR(50),
    fragility_level VARCHAR(20),
    weight_kg FLOAT,
    shipping_distance_km INT
);
```

---

## 📥 Data Ingestion Script

```python
def load_csv_to_db(csv_path, table_name):
    df = pd.read_csv(csv_path)
    df.to_sql(table_name, engine, if_exists='append', index=False)
```

---

## ▶️ How to Run

```bash
python ingest_data.py
python validate_data.py
```

---

## 📤 Output

- PostgreSQL database populated
- Validated eco-friendly packaging dataset
- Ready for Module 2
