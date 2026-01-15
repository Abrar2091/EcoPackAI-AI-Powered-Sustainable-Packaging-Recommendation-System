-- Database initialization script.
-- Creates tables and schema in PostgreSQL database.

CREATE SCHEMA IF NOT EXISTS sustainability;

CREATE TABLE IF NOT EXISTS sustainability.materials (
    material_id SERIAL PRIMARY KEY,
    material_type VARCHAR(200),
    material_subtype VARCHAR(200),
    strength_MPa NUMERIC,
    weight_capacity_kg NUMERIC,
    density_g_per_cm3 NUMERIC,
    cost_per_kg_usd NUMERIC,
    biodegradability_score NUMERIC,
    co2_emission_kgCO2_per_kg NUMERIC,
    recyclability_percentage NUMERIC
);

CREATE TABLE IF NOT EXISTS sustainability.autoliv_materials (
    id SERIAL PRIMARY KEY,
    category_name VARCHAR(200),
    common_products TEXT,
    packaging_requirements TEXT,
    fragility_level INT,
    ideal_material_types TEXT
);
