-- Data import script for PostgreSQL database.

\copy sustainability.materials
FROM 'C:/Users/Lenovo/OneDrive/Desktop/Project2/data/sustainable_materials.csv'
DELIMITER ',' CSV HEADER;

-- FIXED .cv → .csv
\copy sustainability.autoliv_materials
FROM 'C:/Users/Lenovo/OneDrive/Desktop/Project2/data/product_categories_dataset.csv'
DELIMITER ',' CSV HEADER;
