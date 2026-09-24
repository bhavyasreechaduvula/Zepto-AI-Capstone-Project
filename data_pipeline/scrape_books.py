import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

BASE_URL = "https://books.toscrape.com/"
EXCHANGE_RATE = 105.50
TOTAL_PAGES = 5


def get_page(url):
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    response.encoding = "utf-8"
    return BeautifulSoup(response.text, "html.parser")


def clean_price(price_text):
    price_text = price_text.replace("Â", "")
    price_text = price_text.replace("£", "")
    price_text = price_text.strip()

    match = re.search(r"\d+(?:\.\d+)?", price_text)

    if not match:
        raise ValueError(f"Invalid price: {price_text}")

    return float(match.group())


def get_rating(rating_tag):
    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5
    }

    classes = rating_tag.get("class", [])

    if len(classes) < 2:
        return "Unknown", None

    rating_text = classes[1]
    rating_number = rating_map.get(rating_text)

    return rating_text, rating_number


def get_category(book_soup):
    breadcrumb = book_soup.select("ul.breadcrumb li a")

    if len(breadcrumb) >= 3:
        return breadcrumb[-1].get_text(strip=True)

    return "Unknown"


def scrape_all_products():

    books = []

    for page_number in range(1, TOTAL_PAGES + 1):

        if page_number == 1:
            url = BASE_URL + "index.html"
        else:
            url = BASE_URL + f"catalogue/page-{page_number}.html"

        print(f"Reading All Products - page {page_number}...")

        try:
            soup = get_page(url)
        except requests.RequestException as error:
            print(f"Could not read page: {error}")
            continue

        book_cards = soup.select("article.product_pod")

        print(f"Books found on page: {len(book_cards)}")

        for book_card in book_cards:

            try:
                link = book_card.find("h3").find("a")

                # The website gives a relative link such as:
                # catalogue/a-light-in-the-attic_1000/index.html
                href = link.get("href")

                book_url = requests.compat.urljoin(
                    url,
                    href
                )

                book_soup = get_page(book_url)

                # Title
                title = book_soup.find("h1").get_text(strip=True)

                # Price
                price_tag = book_soup.find(
                    "p",
                    class_="price_color"
                )

                price_text = price_tag.get_text(strip=True)
                price_gbp = clean_price(price_text)

                # Rating
                rating_tag = book_soup.find(
                    "p",
                    class_="star-rating"
                )

                star_rating, rating = get_rating(rating_tag)

                # Availability
                availability_tag = book_soup.find(
                    "p",
                    class_="instock availability"
                )

                availability = availability_tag.get_text(
                    " ",
                    strip=True
                )

                in_stock = "In stock" in availability

                # Category
                category = get_category(book_soup)

                books.append({
                    "title": title,
                    "price_gbp": price_gbp,
                    "star_rating": star_rating,
                    "rating": rating,
                    "availability": availability,
                    "in_stock": in_stock,
                    "category": category
                })

            except (
                AttributeError,
                ValueError,
                TypeError,
                KeyError,
                requests.RequestException
            ) as error:

                print(f"Skipped one book: {error}")

    return pd.DataFrame(books)


def clean_data(df):

    df = df.copy()

    df["price_gbp"] = pd.to_numeric(
        df["price_gbp"],
        errors="coerce"
    )

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    df["in_stock"] = df["in_stock"].astype(bool)

    df = df.dropna(
        subset=[
            "title",
            "price_gbp",
            "rating",
            "category"
        ]
    )

    df["rating"] = df["rating"].astype(int)

    df["price_inr"] = (
        df["price_gbp"] * EXCHANGE_RATE
    )

    return df


def main():

    print("Starting book scraping...\n")

    df = scrape_all_products()

    print(
        "\nTotal books scraped:",
        len(df)
    )

    if len(df) >= 60:
        print("60+ books requirement satisfied!")
    else:
        print("WARNING: Less than 60 books were scraped.")

    df = clean_data(df)

    print(
        "Total books after cleaning:",
        len(df)
    )

    print("\nBooks by category:")
    print(df["category"].value_counts())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    output_file = "books_cleaned.csv"

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nSaved cleaned data to: {output_file}"
    )


if __name__ == "__main__":
    main()