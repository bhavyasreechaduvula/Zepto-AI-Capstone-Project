import sqlite3
from pathlib import Path
import pandas as pd


# Project folders
BASE_DIR = Path(__file__).resolve().parent

# books_cleaned.csv is in the main project folder
CSV_FILE = BASE_DIR.parent / "books_cleaned.csv"

# SQLite database will be inside data_pipeline
DB_FILE = BASE_DIR / "books.db"

# Fixed exchange rate required by the project
EXCHANGE_RATE = 105.50


def create_database():

    # Read cleaned CSV
    df = pd.read_csv(CSV_FILE)

    print("Books loaded from CSV:", len(df))

    # Connect to SQLite
    connection = sqlite3.connect(DB_FILE)

    # Enable foreign key support
    connection.execute("PRAGMA foreign_keys = ON")

    cursor = connection.cursor()

    # Remove old tables
    cursor.execute("DROP TABLE IF EXISTS books")
    cursor.execute("DROP TABLE IF EXISTS categories")

    # Create categories table
    cursor.execute("""
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        )
    """)

    # Create books table
    cursor.execute("""
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            rating INTEGER NOT NULL,
            availability TEXT,
            in_stock INTEGER NOT NULL,
            price_inr REAL NOT NULL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
    """)

    # Get unique categories
    categories = sorted(
        df["category"].dropna().unique()
    )

    # Insert categories
    cursor.executemany(
        "INSERT INTO categories (category_name) VALUES (?)",
        [(category,) for category in categories]
    )

    # Get category IDs
    category_rows = cursor.execute(
        "SELECT category_id, category_name FROM categories"
    ).fetchall()

    category_map = {
        category_name: category_id
        for category_id, category_name in category_rows
    }

    # Prepare book records
    book_rows = []

    for _, row in df.iterrows():

        # Calculate INR directly using the required fixed rate
        price_inr = float(row["price_gbp"]) * EXCHANGE_RATE

        book_rows.append((
            row["title"],
            float(row["price_gbp"]),
            int(row["rating"]),
            row["availability"],
            int(bool(row["in_stock"])),
            price_inr,
            category_map[row["category"]]
        ))

    # Insert books
    cursor.executemany("""
        INSERT INTO books (
            title,
            price_gbp,
            rating,
            availability,
            in_stock,
            price_inr,
            category_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, book_rows)

    connection.commit()

    # Count records
    category_count = cursor.execute(
        "SELECT COUNT(*) FROM categories"
    ).fetchone()[0]

    book_count = cursor.execute(
        "SELECT COUNT(*) FROM books"
    ).fetchone()[0]

    print("Categories inserted:", category_count)
    print("Books inserted:", book_count)

    # Test JOIN
    sample = cursor.execute("""
        SELECT
            books.title,
            books.price_gbp,
            books.rating,
            categories.category_name
        FROM books
        JOIN categories
            ON books.category_id = categories.category_id
        LIMIT 5
    """).fetchall()

    print("\nSample database records:")

    for row in sample:
        print(row)

    connection.close()

    print("\nDatabase created successfully!")
    print("Database file:", DB_FILE)


if __name__ == "__main__":
    create_database()