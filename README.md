# Zepto AI Capstone Project

This project contains three modules:

1. Data Pipeline
2. Analytics
3. Support Assistant

---

## Project Structure

```text
Zepto AI Capstone Project/
│
├── data_pipeline/
│   ├── scrape_books.py
│   ├── database.py
│   ├── queries.py
│   ├── books.db
│   └── query_results.txt
│
├── analytics/
│
├── support_assistant/
│
├── books_cleaned.csv
└── README.md

Module 1: Data Pipeline
Objective

The data pipeline collects book information from Books to Scrape, cleans the data, converts GBP prices to INR, stores the data in SQLite, and performs SQL analysis.

Data Collection

The scraper uses Python, requests, and BeautifulSoup.

The first 5 pages of the All Products section are scraped.

There are 20 books per page, giving 100 books in total.

The collected fields include:

title
price_gbp
star_rating
availability
category

The cleaned dataset also contains:

rating
in_stock
price_inr
Data Cleaning

The data is cleaned by:

Converting price to numeric GBP.
Converting star rating to an integer from 1 to 5.
Converting availability to an in_stock boolean.
Removing unnecessary whitespace.
Handling invalid or incomplete records during cleaning.
GBP to INR Conversion

The project uses the required fixed exchange rate:

1 GBP = 105.50 INR

The conversion is:

price_inr = price_gbp * 105.50

The database also calculates INR directly from the GBP price using the same fixed exchange rate.

SQLite Database

The SQLite database is:

data_pipeline/books.db

The database contains two normalized tables.

categories
category_id - Primary Key
category_name
books
book_id - Primary Key
title
price_gbp
rating
availability
in_stock
price_inr
category_id - Foreign Key

The category_id in the books table references the categories table.

SQL Queries

The project demonstrates:

SELECT and WHERE
ORDER BY and LIMIT
DISTINCT
BETWEEN
IN
JOIN

The query results are saved in:

data_pipeline/query_results.txt

The project uses pandas.read_sql() to read SQL query results.

The SQL JOIN result is also reproduced using pandas.merge().

The SQL JOIN and pandas merge results are compared for equivalence.

How to Run Module 1

Install the required packages:

pip install requests beautifulsoup4 pandas

Run the scraper:

python data_pipeline/scrape_books.py

Create the database:

python data_pipeline/database.py

Run the SQL queries:

python data_pipeline/queries.py


# Module 2: Analytics and Machine Learning

## Objective

The objective of Module 2 is to perform exploratory data analysis,
visualization, classification modeling, class imbalance analysis,
Random Forest tuning, and regression analysis using the Titanic dataset.

## Dataset Loading

The Titanic dataset was loaded using Seaborn and saved immediately as:

analytics/titanic.csv

The original Titanic dataset contains 891 rows and 15 columns.

The CSV file is used as an offline fallback for the dataset.

## Data Cleaning

The following missing-value strategy was used:

- `age` had approximately 19.87% missing values, so median imputation was used.
- `embarked` had less than 5% missing values, so affected rows were dropped.
- `embark_town` had less than 5% missing values, so affected rows were dropped.
- `deck` had approximately 77.22% missing values, so the column was dropped because of very high missingness.

After cleaning:

- Rows: 889
- Columns: 14
- No missing values remained.

The cleaned dataset was saved as:

analytics/titanic_cleaned.csv

## Univariate Analysis

The following analyses were performed:

- Age histogram
- Age boxplot
- Fare histogram
- Fare boxplot
- IQR outlier detection for Age and Fare
- Fare mean
- Fare median
- Fare mode
- Fare skewness

Fare was found to be right-skewed.

The relationship between the three measures was:

Mean > Median > Mode

Generated outputs include:

- `age_histogram.png`
- `age_boxplot.png`
- `fare_histogram.png`
- `fare_boxplot.png`
- `univariate_summary.txt`

## Bivariate Analysis

Survival rates were calculated for:

- Sex
- Passenger class
- Sex and passenger class

A correlation matrix was created using exactly these six columns:

- survived
- pclass
- age
- sibsp
- parch
- fare

The two strongest absolute correlations were:

- `pclass` and `fare`
- `sibsp` and `parch`

Generated outputs include:

- `survival_rates.csv`
- `correlation_matrix.csv`
- `correlation_heatmap.png`
- `top_two_correlations.csv`

## Multivariate Analysis

Four different multivariate visualizations were created:

1. Survival rate by sex and passenger class
2. Age distribution by survival and sex
3. Fare distribution by passenger class and survival
4. Family size, passenger class and survival

Generated outputs include:

- `survival_sex_pclass.png`
- `age_survival_sex.png`
- `fare_pclass_survival.png`
- `family_size_survival_class.png`
- `multivariate_interpretations.txt`

## Z-Score Standardization

Z-score standardization was performed on:

- `age`
- `fare`

The standardized variables have approximately:

- Mean = 0
- Standard deviation = 1

Generated outputs:

- `titanic_zscore.csv`
- `zscore_summary.csv`

## Classification Modeling

Three classification models were trained using the same stratified
train-test split:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The preprocessing pipeline included:

- Median imputation for numerical variables
- Most-frequent imputation for categorical variables
- One-hot encoding for categorical variables
- StandardScaler for numerical variables

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

The comparison results were saved as:

`classification_comparison.csv`

Additional outputs include:

- `confusion_matrix_logistic_regression.png`
- `confusion_matrix_decision_tree.png`
- `confusion_matrix_random_forest.png`
- `roc_curves.png`
- `decision_tree.png`

## Class Imbalance Analysis

Three approaches were compared:

1. Baseline Logistic Regression
2. Logistic Regression with `class_weight="balanced"`
3. Logistic Regression with SMOTE

SMOTE was applied only to the training data.

The test data was kept unchanged for evaluation.

Generated outputs:

- `imbalance_comparison.csv`
- `imbalance_comparison.png`
- `imbalance_conclusion.txt`

## Random Forest Hyperparameter Tuning

GridSearchCV was used to tune the following parameters:

- `n_estimators`
- `max_depth`
- `max_features`

The Random Forest estimator used:

`oob_score=True`

The best parameters and OOB score were saved in:

`random_forest_tuning.txt`

The tuned Random Forest pipeline was saved as:

`best_random_forest_pipeline.joblib`

## Regression Analysis

A multivariate Linear Regression model was used to predict passenger
fare from other passenger features.

The following metrics were calculated:

- MAE
- RMSE
- R²
- Adjusted R²

A residual plot was created to inspect possible heteroscedasticity.

Generated outputs:

- `regression_metrics.csv`
- `regression_summary.txt`
- `regression_residual_plot.png`

## Final Model Comparison

Classification and regression metrics were reported separately.

Classification metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Regression metrics:

- MAE
- RMSE
- R²
- Adjusted R²

The classification models were compared using the test-set metrics.

## Saved Classification Pipeline

The complete fitted classification pipeline was saved using Joblib:

`final_classifier_pipeline.joblib`

The saved pipeline was reloaded and tested using a raw passenger input
to confirm that the complete preprocessing and prediction workflow works.

## How to Run Module 2

From the project root directory, activate the virtual environment:

```powershell
.venv\Scripts\activate

## Module 3 – AI Support Assistant

### Architecture

The support assistant uses a Retrieval-Augmented Generation (RAG) pipeline:

Documents → Chunking → all-MiniLM-L6-v2 Embeddings → ChromaDB → Retrieval → LangGraph → Answer Generation → Pydantic Validation → FastAPI

### Components

- `support_assistant/docs/` – contains 8 Zepto policy documents.
- `support_assistant/ingest.py` – loads policy documents, creates chunks, generates embeddings, and stores them in ChromaDB.
- `support_assistant/chroma_db/` – persistent ChromaDB vector store containing the policy embeddings.
- `support_assistant/graph.py` – handles intent classification, retrieval, answer generation, validation, and LangGraph routing.
- `support_assistant/prompts.py` – contains the structured prompt with role, context, task, format, length, negative constraint, and few-shot example.
- `support_assistant/schemas.py` – defines the Pydantic response schema with `answer`, `sources`, and `confidence`.
- `support_assistant/api.py` – provides the FastAPI `/ask` endpoint.
- `Dockerfile` – contains the container configuration.

### RAG Data Flow

1. Policy documents are loaded from `support_assistant/docs/`.
2. Documents are split into smaller chunks.
3. Chunks are converted into embeddings using `all-MiniLM-L6-v2`.
4. Embeddings are stored in ChromaDB.
5. A user query is converted into an embedding.
6. ChromaDB retrieves the most relevant policy chunks.
7. LangGraph routes the query and generates the answer.
8. The generated response is validated using Pydantic.
9. FastAPI returns the final JSON response.

### LangGraph Flow

The LangGraph workflow contains the following nodes:

- `classify_intent`
- `retrieve_context`
- `generate_answer`
- `validate_output`
- `direct_answer`

Policy questions are sent through retrieval and answer generation. General questions are handled by the direct-answer branch.

### Structured Prompt

The prompt contains:

- Role
- Context
- Task
- Format
- Length
- Negative constraint
- Few-shot example

The negative constraint prevents the assistant from inventing policy information that is not present in the retrieved context.

### Pydantic Response

The API response follows this schema:

```json
{
  "answer": "string",
  "sources": ["string"],
  "confidence": 1.0
}
### Testing

The FastAPI `/ask` endpoint was tested successfully with delivery-fee and order-cancellation queries, both returning HTTP 200 responses.