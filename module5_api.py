# module5_api.py
# Flask Backend API for EcoPackAI
# - REST APIs: products, materials, recommendations, environmental score
# - PostgreSQL connection (user: postgres, password: pass, db: postgres)
# - Secure endpoints via API key
# - Consistent JSON responses
# - Auto-add missing columns on startup

import os
import joblib
import pandas as pd
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps

from sqlalchemy import (
    create_engine, Column, Integer, Float, String, DateTime, JSON, text, inspect
)
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
API_KEY = os.getenv("API_KEY", "super")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:pass@localhost:5432/postgres"
)
RF_COST_MODEL_PATH = os.getenv("RF_COST_MODEL_PATH", "rf_cost_model.pkl")
XGB_CO2_MODEL_PATH = os.getenv("XGB_CO2_MODEL_PATH", "xgb_co2_model.pkl")

COST_WEIGHT = float(os.getenv("COST_WEIGHT", "0.5"))
CO2_WEIGHT = float(os.getenv("CO2_WEIGHT", "0.5"))

FEATURE_COLS = [
    "strength_MPa",
    "weight_capacity_kg",
    "density_g_per_cm3",
    "biodegradability_score",
    "recyclability_percentage",
    "material_suitability_score",
]

# -----------------------------------------------------------------------------
# App and DB setup
# -----------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = scoped_session(sessionmaker(bind=engine, autocommit=False, autoflush=False))
Base = declarative_base()

# -----------------------------------------------------------------------------
# Database models (include all fields)
# -----------------------------------------------------------------------------
class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    description = Column(String(1000), nullable=True)
    max_cost_per_kg_usd = Column(Float, nullable=True)
    max_co2_emission_kgCO2_per_kg = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class MaterialCandidate(Base):
    __tablename__ = "material_candidates"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    material_type = Column(String(200), nullable=True)
    material_subtype = Column(String(200), nullable=True)
    strength_MPa = Column(Float, nullable=False)
    weight_capacity_kg = Column(Float, nullable=False)
    density_g_per_cm3 = Column(Float, nullable=False)
    biodegradability_score = Column(Float, nullable=False)
    recyclability_percentage = Column(Float, nullable=False)
    material_suitability_score = Column(Float, nullable=False)
    predicted_cost_per_kg_usd = Column(Float, nullable=True)
    predicted_co2_kgCO2_per_kg = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class RecommendationRecord(Base):
    __tablename__ = "recommendation_records"
    id = Column(Integer, primary_key=True, autoincrement=True)
    material_type = Column(String(200), nullable=True)
    material_subtype = Column(String(200), nullable=True)
    product_id = Column(Integer, nullable=True)
    inputs_snapshot = Column(JSON, nullable=True)
    results_snapshot = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class EnvironmentalScore(Base):
    __tablename__ = "environmental_scores"
    id = Column(Integer, primary_key=True, autoincrement=True)
    material_id = Column(Integer, nullable=True)
    material_name = Column(String(200), nullable=False)
    co2_kgCO2_per_kg = Column(Float, nullable=False)
    recyclability_percentage = Column(Float, nullable=False)
    biodegradability_score = Column(Float, nullable=False)
    environmental_score = Column(Float, nullable=False)
    formula_version = Column(String(50), default="v1.0")
    created_at = Column(DateTime, default=datetime.utcnow)

# -----------------------------------------------------------------------------
# Ensure schema exists and add missing columns if necessary
# -----------------------------------------------------------------------------
def ensure_schema_and_columns():
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    with engine.connect() as conn:
        # products: ensure description, max_cost_per_kg_usd, max_co2_emission_kgCO2_per_kg exist
        if "products" in inspector.get_table_names():
            cols = {c["name"] for c in inspector.get_columns("products")}
            if "description" not in cols:
                conn.execute(text('ALTER TABLE products ADD COLUMN description VARCHAR(1000);'))
            if "max_cost_per_kg_usd" not in cols:
                conn.execute(text('ALTER TABLE products ADD COLUMN max_cost_per_kg_usd FLOAT;'))
            if "max_co2_emission_kgCO2_per_kg" not in cols:
                conn.execute(text('ALTER TABLE products ADD COLUMN max_co2_emission_kgCO2_per_kg FLOAT;'))
        # material_candidates: ensure material_type and material_subtype exist
        if "material_candidates" in inspector.get_table_names():
            cols = {c["name"] for c in inspector.get_columns("material_candidates")}
            if "material_type" not in cols:
                conn.execute(text('ALTER TABLE material_candidates ADD COLUMN material_type VARCHAR(200);'))
            if "material_subtype" not in cols:
                conn.execute(text('ALTER TABLE material_candidates ADD COLUMN material_subtype VARCHAR(200);'))
        conn.commit()

# Run schema ensure at import time
try:
    ensure_schema_and_columns()
except Exception as e:
    print("Schema ensure failed:", e)

# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def try_load_model(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

rf_cost_model = try_load_model(RF_COST_MODEL_PATH)
xgb_co2_model = try_load_model(XGB_CO2_MODEL_PATH)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def json_success(data, status_code=200):
    return jsonify({"status": "success", "data": data}), status_code

def json_error(message, code="bad_request", status_code=400, details=None):
    return jsonify({"status": "error", "error": {"code": code, "message": message, "details": details}}), status_code

def require_api_key(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        header_key = request.headers.get("X-API-Key")
        if not header_key or header_key != API_KEY:
            return jsonify({
                "status": "error",
                "error": {"code": "unauthorized", "message": "Missing or invalid API key."}
            }), 401
        return func(*args, **kwargs)
    return wrapper

def validate_material_payload(m: dict):
    missing = [k for k in FEATURE_COLS if k not in m]
    if missing:
        raise ValueError(f"Missing required feature(s): {', '.join(missing)}")

    cleaned = {}
    for k in FEATURE_COLS:
        try:
            cleaned[k] = float(m[k])
        except (TypeError, ValueError):
            raise ValueError(f"Feature '{k}' must be numeric.")

    name = m.get("name", "Unnamed material")

    # ✅ FIX 2: make optional with defaults
    mat_type = m.get("material_type", "unknown")
    mat_subtype = m.get("material_subtype", "unknown")

    return name, mat_type, mat_subtype, cleaned


def predict_cost_and_co2(materials):
    if rf_cost_model is None or xgb_co2_model is None:
        raise RuntimeError("Models are not loaded. Ensure rf_cost_model.pkl and xgb_co2_model.pkl are available.")
    X = pd.DataFrame(materials, columns=FEATURE_COLS)
    cost_preds = rf_cost_model.predict(X)
    co2_preds = xgb_co2_model.predict(X)
    out = []
    for i, mat in enumerate(materials):
        o = dict(mat)
        o["predicted_cost_per_kg_usd"] = float(cost_preds[i])
        o["predicted_co2_kgCO2_per_kg"] = float(co2_preds[i])
        out.append(o)
    return out

def rank_materials(predicted_materials, cost_weight=COST_WEIGHT, co2_weight=CO2_WEIGHT):
    ranked = []
    for m in predicted_materials:
        cost = m["predicted_cost_per_kg_usd"]
        co2 = m["predicted_co2_kgCO2_per_kg"]
        score = (cost_weight * cost) + (co2_weight * co2)
        ranked.append({**m, "material_rank_score": float(score)})
    ranked.sort(key=lambda x: x["material_rank_score"])
    return ranked

def compute_environmental_score(co2, recyclability_pct, biodegradability_score):
    norm_co2 = min(max(float(co2), 0.0), 10.0) / 10.0
    r = min(max(float(recyclability_pct), 0.0), 100.0) / 100.0
    b = min(max(float(biodegradability_score), 0.0), 10.0) / 10.0
    score = 100.0 * (0.6 * (1.0 - norm_co2) + 0.3 * r + 0.1 * b)
    return round(score, 2)

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return json_success({"service": "EcoPackAI Backend", "status": "ok"})

# Products
@app.route("/api/products", methods=["POST"])
@require_api_key
def create_product():
    payload = request.get_json(silent=True) or {}
    name = payload.get("name")
    description = payload.get("description")
    max_cost = payload.get("max_cost_per_kg_usd")
    max_co2 = payload.get("max_co2_emission_kgCO2_per_kg")
    if not name:
        return json_error("Field 'name' is required.", code="validation_error")
    try:
        db = SessionLocal()
        p = Product(
            name=name,
            description=description,
            max_cost_per_kg_usd=float(max_cost) if max_cost is not None else None,
            max_co2_emission_kgCO2_per_kg=float(max_co2) if max_co2 is not None else None,
        )
        db.add(p)
        db.commit()
        db.refresh(p)
        return json_success({"id": p.id, "name": p.name}, status_code=201)
    except Exception as e:
        return json_error("Failed to create product.", code="db_error", details=str(e), status_code=500)
    finally:
        SessionLocal.remove()

@app.route("/api/products", methods=["GET"])
@require_api_key
def list_products():
    try:
        db = SessionLocal()
        products = db.query(Product).order_by(Product.id.asc()).all()
        out = []
        for p in products:
            out.append({
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "max_cost_per_kg_usd": p.max_cost_per_kg_usd,
                "max_co2_emission_kgCO2_per_kg": p.max_co2_emission_kgCO2_per_kg,
                "created_at": p.created_at.isoformat()
            })
        return json_success({"products": out})
    except Exception as e:
        return json_error("Failed to list products.", code="db_error", details=str(e), status_code=500)
    finally:
        SessionLocal.remove()

# Materials
@app.route("/api/materials", methods=["POST"])
@require_api_key
def upsert_materials():
    payload = request.get_json(silent=True) or {}
    materials = payload.get("materials", None)

# treat empty list / empty dict as NOT provided
    if materials in (None, [], {}):
       materials = None

    if not isinstance(materials, list) or not materials:
        return json_error("Field 'materials' must be a non-empty list.", code="validation_error")
    try:
        db = SessionLocal()
        inserted = []
        for m in materials:
            name, mat_type, mat_subtype, cleaned = validate_material_payload(m)
            obj = MaterialCandidate(
                name=name,
                material_type=mat_type,
                material_subtype=mat_subtype,
                strength_MPa=cleaned["strength_MPa"],
                weight_capacity_kg=cleaned["weight_capacity_kg"],
                density_g_per_cm3=cleaned["density_g_per_cm3"],
                biodegradability_score=cleaned["biodegradability_score"],
                recyclability_percentage=cleaned["recyclability_percentage"],
                material_suitability_score=cleaned["material_suitability_score"],
            )
            db.add(obj)
            db.flush()
            inserted.append({"id": obj.id, "name": obj.name})
        db.commit()
        return json_success({"inserted": inserted}, status_code=201)
    except ValueError as ve:
        return json_error(str(ve), code="validation_error")
    except Exception as e:
        return json_error("Failed to upsert materials.", code="db_error", details=str(e), status_code=500)
    finally:
        SessionLocal.remove()

@app.route("/api/materials", methods=["GET"])
@require_api_key
def list_materials():
    try:
        db = SessionLocal()
        mats = db.query(MaterialCandidate).order_by(MaterialCandidate.id.desc()).all()
        out = []
        for m in mats:
            out.append({
                "id": m.id,
                "name": m.name,
                "material_type": m.material_type,
                "material_subtype": m.material_subtype,
                "strength_MPa": m.strength_MPa,
                "weight_capacity_kg": m.weight_capacity_kg,
                "density_g_per_cm3": m.density_g_per_cm3,
                "biodegradability_score": m.biodegradability_score,
                "recyclability_percentage": m.recyclability_percentage,
                "material_suitability_score": m.material_suitability_score,
                "predicted_cost_per_kg_usd": m.predicted_cost_per_kg_usd,
                "predicted_co2_kgCO2_per_kg": m.predicted_co2_kgCO2_per_kg,
                "created_at": m.created_at.isoformat(),
            })
        return json_success({"materials": out})
    except Exception as e:
        return json_error("Failed to list materials.", code="db_error", details=str(e), status_code=500)
    finally:
        SessionLocal.remove()

# AI Material Recommendation
@app.route("/api/recommendations/materials", methods=["POST"])
@require_api_key
def recommend_materials():
    payload = request.get_json(silent=True) or {}
    materials = payload.get("materials")
    product_id = payload.get("product_id")
    weights = payload.get("weights", {})
    cost_w = float(weights.get("cost", COST_WEIGHT))
    co2_w = float(weights.get("co2", CO2_WEIGHT))
    try:
        db = SessionLocal()
        candidate_dicts = []
        if isinstance(materials, list) and materials:
            for m in materials:
                name, mat_type, mat_subtype, cleaned = validate_material_payload(m)
                candidate_dicts.append({"name": name, **cleaned})
        else:
            mats = db.query(MaterialCandidate).all()
            if not mats:
                return json_error("No materials available for recommendation.", code="no_materials", status_code=404)
            for m in mats:
                candidate_dicts.append({
                    "name": m.name,
                    "strength_MPa": m.strength_MPa,
                    "weight_capacity_kg": m.weight_capacity_kg,
                    "density_g_per_cm3": m.density_g_per_cm3,
                    "biodegradability_score": m.biodegradability_score,
                    "recyclability_percentage": m.recyclability_percentage,
                    "material_suitability_score": m.material_suitability_score,
                })

        preds = predict_cost_and_co2(candidate_dicts)
        ranked = rank_materials(preds, cost_weight=cost_w, co2_weight=co2_w)

        record = RecommendationRecord(
            product_id=int(product_id) if product_id is not None else None,
            inputs_snapshot=payload,
            results_snapshot={"ranked": ranked}
        )
        db.add(record)

        # Cache predictions back to DB by name
        name_to_pred = {r["name"]: r for r in ranked}
        for m in db.query(MaterialCandidate).all():
            if m.name in name_to_pred:
                r = name_to_pred[m.name]
                m.predicted_cost_per_kg_usd = r["predicted_cost_per_kg_usd"]
                m.predicted_co2_kgCO2_per_kg = r["predicted_co2_kgCO2_per_kg"]

        db.commit()
        return json_success({
            "weights_used": {"cost": cost_w, "co2": co2_w},
            "recommendations": ranked
        })
    except ValueError as ve:
        return json_error(str(ve), code="validation_error")
    except RuntimeError as re:
        return json_error(str(re), code="model_not_loaded", status_code=500)
    except Exception as e:
        return json_error("Failed to generate recommendations.", code="server_error", details=str(e), status_code=500)
    finally:
        SessionLocal.remove()

# Environmental Score
@app.route("/api/score/environmental", methods=["POST"])
@require_api_key
def environmental_score():
    payload = request.get_json(silent=True) or {}
    material_name = payload.get("material_name")
    co2 = payload.get("co2_kgCO2_per_kg")
    recyclability_pct = payload.get("recyclability_percentage")
    biodegradability_score = payload.get("biodegradability_score")
    if not material_name:
        return json_error("Field 'material_name' is required.", code="validation_error")
    try:
        co2 = float(co2)
        recyclability_pct = float(recyclability_pct)
        biodegradability_score = float(biodegradability_score)
    except (TypeError, ValueError):
        return json_error("Numeric fields are invalid.", code="validation_error")
    try:
        score = compute_environmental_score(co2, recyclability_pct, biodegradability_score)
        db = SessionLocal()
        material = db.query(MaterialCandidate).filter(MaterialCandidate.name == material_name).first()
        record = EnvironmentalScore(
            material_id=material.id if material else None,
            material_name=material_name,
            co2_kgCO2_per_kg=co2,
            recyclability_percentage=recyclability_pct,
            biodegradability_score=biodegradability_score,
            environmental_score=score,
            formula_version="v1.0"
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return json_success({
            "material_name": material_name,
            "environmental_score": score,
            "formula_version": "v1.0",
            "record_id": record.id
        })
    except Exception as e:
        return json_error("Failed to compute environmental score.", code="server_error", details=str(e), status_code=500)
    finally:
        SessionLocal.remove()

@app.route("/")
def home():
    return "EcoPackAI Backend is running"



from flask_sqlalchemy import SQLAlchemy

# 🔹 CONNECT TO EXISTING POSTGRES DATABASE
app.config["SQLALCHEMY_DATABASE_URI"] = (
    "postgresql+psycopg2://postgres:pass@localhost:5432/postgres"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

@app.route("/api/module7/bi-data", methods=["GET"])
def module7_bi_data():
    try:
        db = SessionLocal()
        materials = db.query(MaterialCandidate).all()

        output = []
        for m in materials:
            if m.predicted_cost_per_kg_usd is None or m.predicted_co2_kgCO2_per_kg is None:
                continue  # skip non-predicted rows

            output.append({
                "name": m.name,
                "predicted_cost": m.predicted_cost_per_kg_usd,
                "predicted_co2": m.predicted_co2_kgCO2_per_kg,
                "baseline_cost": 0.60,   # fixed baseline (can be config)
                "baseline_co2": 0.80,
                "created_at": m.created_at.isoformat()
            })

        return json_success(output)
    except Exception as e:
        return json_error(
            "Failed to load BI dashboard data",
            code="bi_error",
            details=str(e),
            status_code=500
        )
    finally:
        SessionLocal.remove()

if __name__ == "__main__":
    ensure_schema_and_columns()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
