# 📦 EcoPackAI  
## Module 2: Data Cleaning & Feature Engineering  
(Sustainability Metrics Preparation)

---

## 📌 Module Overview

Module 2 converts raw data collected in Module 1 into a clean, consistent, and machine-learning–ready dataset.  
This module focuses on improving data quality and creating domain-specific sustainability features required for accurate cost and CO₂ prediction.

---

## 🎯 Objectives

- Handle missing, inconsistent, and duplicate data  
- Normalize numerical features  
- Encode categorical attributes  
- Engineer sustainability metrics:
  - CO₂ Impact Index
  - Cost Efficiency Index
  - Material Suitability Score
- Validate cleaned dataset

---

## 🔄 Input & Output

### Input
- PostgreSQL tables:
  - materials
  - products

### Output
- cleaned_materials.csv
- engineered_features.csv
- Dataset ready for ML (Module 3)

---

## 🗂 Folder Structure

```
EcoPackAI/
└── module2_data_cleaning/
    ├── preprocessing/
    │   ├── clean_data.py
    │   ├── feature_engineering.py
    │   └── encode_features.py
    ├── outputs/
    │   ├── cleaned_materials.csv
    │   └── engineered_features.csv
    └── README.md
```

---

## 🛠 Packages Used

| Package | Version |
|------|--------|
| python | 3.10 |
| pandas | 2.1.1 |
| numpy | 1.26.2 |
| scikit-learn | 1.3.2 |
| sqlalchemy | 2.0.23 |

---

## ⚙️ Installation

```bash
pip install pandas==2.1.1 numpy==1.26.2 scikit-learn==1.3.2 sqlalchemy==2.0.23
```

---

## 🧹 Step 1: Data Cleaning

File: `clean_data.py`

```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://postgres:password@localhost:5432/ecopackai_db")

def clean_material_data():
    df = pd.read_sql("SELECT * FROM materials", engine)
    df.drop_duplicates(inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.to_csv("outputs/cleaned_materials.csv", index=False)
    return df
```

### Functions Used
- `read_sql()` – Read data from PostgreSQL
- `drop_duplicates()` – Remove duplicate records
- `fillna()` – Handle missing values
- `to_csv()` – Save cleaned dataset

---

## ⚖️ Step 2: Feature Engineering

File: `feature_engineering.py`

### Engineered Metrics

- **CO₂ Impact Index**
```
(co2_emission_score / max_co2) * 100
```

- **Cost Efficiency Index**
```
strength_rating / cost_per_unit
```

- **Material Suitability Score**
```
(biodegradability + recyclability + strength) / co2_emission_score
```

```python
def engineer_features(df):
    df['co2_impact_index'] = (
        df['co2_emission_score'] / df['co2_emission_score'].max()
    ) * 100

    df['cost_efficiency_index'] = (
        df['strength_rating'] / df['cost_per_unit']
    )

    df['material_suitability_score'] = (
        df['biodegradability_score'] +
        df['recyclability_percent'] +
        df['strength_rating']
    ) / df['co2_emission_score']

    df.to_csv("outputs/engineered_features.csv", index=False)
    return df
```

---

## 🔠 Step 3: Encoding Categorical Features

File: `encode_features.py`

```python
from sklearn.preprocessing import LabelEncoder

def encode_categorical(df):
    encoder = LabelEncoder()
    df['material_type_encoded'] = encoder.fit_transform(df['material_type'])
    return df
```

---

## ▶️ How to Run Module 2

```bash
python preprocessing/clean_data.py
python preprocessing/feature_engineering.py
python preprocessing/encode_features.py
```

---

## 📤 Outputs

| File | Description |
|----|------------|
| cleaned_materials.csv | Cleaned dataset |
| engineered_features.csv | Feature-engineered dataset |

---

## 📈 Best Practices

- Remove duplicates before feature creation  
- Normalize and validate feature ranges  
- Keep preprocessing modular  
- Save intermediate outputs  

---

## 🔗 Dependency

Output of this module is used by:
➡ Module 3: Machine Learning Dataset Preparation

---

## ✅ Module 2 Status

✔ Data cleaned  
✔ Sustainability features engineered  
✔ Dataset ready for ML
