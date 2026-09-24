import sqlite3
from pathlib import Path
import pandas as pd


# File locations
BASE_DIR = Path(__file__).resolve().parent
DB_FILE = BASE_DIR / "books.db"
OUTPUT_FILE = BASE_DIR / "query_results.txt"


def run_queries():

    connection = sqlite3.connect(DB_FILE)

    # ---------------------------------------------------------
    # QUERY 1: SELECT + WHERE
    # ---------------------------------------------------------
    query_1 = """
    SELECT title, price_gbp, rating
    FROM books
    WHERE rating >= 4
    """

    result_1 = pd.read_sql(query_1, connection)

    # ---------------------------------------------------------
    # QUERY 2: ORDER BY + LIMIT
    # ---------------------------------------------------------
    query_2 = """
    SELECT title, price_gbp
    FROM books
    ORDER BY price_gbp DESC
    LIMIT 10
    """

    result_2 = pd.read_sql(query_2, connection)

    # ---------------------------------------------------------
    # QUERY 3: DISTINCT
    # ---------------------------------------------------------
    query_3 = """
    SELECT DISTINCT category_name
    FROM categories
    ORDER BY category_name
    """

    result_3 = pd.read_sql(query_3, connection)

    # ---------------------------------------------------------
    # QUERY 4: BETWEEN
    # ---------------------------------------------------------
    query_4 = """
    SELECT title, price_gbp, rating
    FROM books
    WHERE price_gbp BETWEEN 20 AND 40
    ORDER BY price_gbp
    """

    result_4 = pd.read_sql(query_4, connection)

    # ---------------------------------------------------------
    # QUERY 5: IN
    # ---------------------------------------------------------
    query_5 = """
    SELECT title, rating, category_id
    FROM books
    WHERE rating IN (4, 5)
    ORDER BY rating DESC
    """

    result_5 = pd.read_sql(query_5, connection)

    # ---------------------------------------------------------
    # QUERY 6: JOIN
    # ---------------------------------------------------------
    query_6 = """
    SELECT
        books.title,
        books.price_gbp,
        books.rating,
        categories.category_name
    FROM books
    JOIN categories
        ON books.category_id = categories.category_id
    ORDER BY books.price_gbp DESC
    LIMIT 10
    """

    result_6 = pd.read_sql(query_6, connection)

    # ---------------------------------------------------------
    # Read JOIN data separately for pandas merge comparison
    # ---------------------------------------------------------

    books_query = """
    SELECT
        book_id,
        title,
        price_gbp,
        rating,
        category_id
    FROM books
    """

    categories_query = """
    SELECT
        category_id,
        category_name
    FROM categories
    """

    books_df = pd.read_sql(books_query, connection)
    categories_df = pd.read_sql(categories_query, connection)

    # Reproduce SQL JOIN using pandas merge
    merge_result = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner"
    )

    # Select same columns as SQL JOIN
    merge_result = merge_result[
        [
            "title",
            "price_gbp",
            "rating",
            "category_name"
        ]
    ]

    merge_result = merge_result.sort_values(
        "price_gbp",
        ascending=False
    ).head(10)

    # Check whether SQL JOIN and pandas merge give same result
    sql_join_compare = result_6.reset_index(drop=True)
    pandas_merge_compare = merge_result.reset_index(drop=True)

    join_equivalent = sql_join_compare.equals(
        pandas_merge_compare
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print("\n========== QUERY 1: SELECT + WHERE ==========")
    print(result_1.to_string(index=False))

    print("\n========== QUERY 2: ORDER BY + LIMIT ==========")
    print(result_2.to_string(index=False))

    print("\n========== QUERY 3: DISTINCT ==========")
    print(result_3.to_string(index=False))

    print("\n========== QUERY 4: BETWEEN ==========")
    print(result_4.to_string(index=False))

    print("\n========== QUERY 5: IN ==========")
    print(result_5.to_string(index=False))

    print("\n========== QUERY 6: SQL JOIN ==========")
    print(result_6.to_string(index=False))

    print("\n========== PANDAS MERGE ==========")
    print(merge_result.to_string(index=False))

    print("\n========== JOIN EQUIVALENCE ==========")
    print("SQL JOIN == pandas merge:", join_equivalent)

    # ---------------------------------------------------------
    # Save query strings and outputs
    # ---------------------------------------------------------

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        file.write("BOOKS DATABASE QUERY RESULTS\n")
        file.write("=" * 70 + "\n\n")

        file.write("QUERY 1: SELECT + WHERE\n")
        file.write(query_1.strip() + "\n\n")
        file.write(result_1.to_string(index=False))
        file.write("\n\n" + "=" * 70 + "\n\n")

        file.write("QUERY 2: ORDER BY + LIMIT\n")
        file.write(query_2.strip() + "\n\n")
        file.write(result_2.to_string(index=False))
        file.write("\n\n" + "=" * 70 + "\n\n")

        file.write("QUERY 3: DISTINCT\n")
        file.write(query_3.strip() + "\n\n")
        file.write(result_3.to_string(index=False))
        file.write("\n\n" + "=" * 70 + "\n\n")

        file.write("QUERY 4: BETWEEN\n")
        file.write(query_4.strip() + "\n\n")
        file.write(result_4.to_string(index=False))
        file.write("\n\n" + "=" * 70 + "\n\n")

        file.write("QUERY 5: IN\n")
        file.write(query_5.strip() + "\n\n")
        file.write(result_5.to_string(index=False))
        file.write("\n\n" + "=" * 70 + "\n\n")

        file.write("QUERY 6: JOIN\n")
        file.write(query_6.strip() + "\n\n")
        file.write(result_6.to_string(index=False))
        file.write("\n\n" + "=" * 70 + "\n\n")

        file.write("PANDAS MERGE RESULT\n")
        file.write(merge_result.to_string(index=False))
        file.write("\n\n")

        file.write("JOIN EQUIVALENCE\n")
        file.write(
            f"SQL JOIN == pandas merge: {join_equivalent}\n"
        )

    connection.close()

    print("\nQuery results saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    run_queries()