# load_data.py
# This script loads the dataset from a SQLite database and returns it as a pandas DataFrame.

import time
import sqlite3
import pandas as pd
from pathlib import Path


def load_data(db_path, db_table_name):
    """
    Load the dataset from the specified SQLite database path and table name.
    
    Args:
        db_path (str): Path to the SQLite database file.
        db_table_name (str): Name of the table to load data from.
    
    Returns:
        pandas.DataFrame: The dataset loaded into a DataFrame.
    
    Raises:
        FileNotFoundError: If the database file is not found.
        ValueError: If the specified table does not exist in the database.
        sqlite3.Error: If there's an error connecting to or querying the database.
    """
    print("📥 Starting data loading process...")
    start_time = time.time()

    db_path = validate_database_path(db_path)
    conn = connect_to_database(db_path)
    validate_table_exists(conn, db_table_name)

    # Load data from the specified table
    data = query_table(conn, db_table_name)
    conn.close()

    end_time = time.time() 
    elapsed_time = end_time - start_time
    print(f"\n✅ {len(data):,} records successfully loaded in {elapsed_time:.2f} seconds!")
    return data


def validate_database_path(db_path):
    """
    Validate that the database file exists at the specified path.

    Args:
        db_path (str): Path to the SQLite database file.

    Returns:
        Path: Validated Path object for the database file.

    Raises:
        FileNotFoundError: If the database file is not found.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"❌ Database file not found at {db_path}. Please ensure the dataset is in the correct location.")
    print(f"   └── Database file found at: {db_path}")
    return db_path


def connect_to_database(db_path):
    """
    Connect to the SQLite database.

    Args:
        db_path (Path): Path to the SQLite database file.

    Returns:
        sqlite3.Connection: Connection object to the SQLite database.

    Raises:
        sqlite3.Error: If there's an error connecting to the database.
    """
    try:
        print(f"   └── Connecting to SQLite database at path: {db_path}...")
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        raise sqlite3.Error(f"❌ Failed to connect to the database: {e}") from e


def validate_table_exists(conn, db_table_name):
    """
    Validate that the specified table exists in the database.

    Args:
        conn (sqlite3.Connection): Connection object to the SQLite database.
        db_table_name (str): Name of the table to validate.

    Raises:
        ValueError: If the specified table does not exist in the database.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]
    print(f"   └── Checking if table '{db_table_name}' exists in the database...")
    if db_table_name not in tables:
        raise ValueError(f"❌ Table '{db_table_name}' not found in the database. Available tables: {tables}")


def query_table(conn, db_table_name):
    """
    Query all data from the specified table in the database.

    Args:
        conn (sqlite3.Connection): Connection object to the SQLite database.
        db_table_name (str): Name of the table to query.

    Returns:
        pandas.DataFrame: Data loaded from the specified table.
    """
    print(f"   └── Loading data from table '{db_table_name}' into a pandas DataFrame...")
    query = f"SELECT * FROM {db_table_name}"
    try:
        data = pd.read_sql_query(query, conn)
        return data
    except Exception as e:
        raise Exception(f"❌ Error querying data from table '{db_table_name}': {e}") from e