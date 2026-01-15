"""
Database configuration module for PostgreSQL connection.
Handles database connection parameters and provides connection utilities.
"""

import psycopg2

def get_connection():
    return psycopg2.connect(
        host="localhost",
        port="5432",         # Change if you changed Postgres port
        database="postgres",
        user="postgres",
        password="pass"
    )
