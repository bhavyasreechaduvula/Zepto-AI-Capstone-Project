import os
import warnings

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE_DIR, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# STEP 1: LOAD TITANIC DATA ONCE + CLEAN DATA
# ============================================================

def load_and_clean_data():

    print("\n" + "=" * 70)
    print("STEP 1: LOAD AND CLEAN TITANIC DATA")
    print("=" * 70)

    # Load Titanic only once
    df = sns.load_dataset("titanic")

    print("Titanic dataset loaded successfully!")
    print("Original shape:", df.shape)

    # Save immediately as offline fallback
    titanic_csv = os.path.join(BASE_DIR, "titanic.csv")
    df.to_csv(titanic_csv, index=False)

    print("Original Titanic CSV saved to:")
    print(titanic_csv)

    # --------------------------------------------------------
    # PROFILE
    # --------------------------------------------------------

    profile_file = os.path.join(BASE_DIR, "data_profile.txt")

    with open(profile_file, "w", encoding="utf-8") as file:

        file.write("TITANIC DATA PROFILE\n")
        file.write("=" * 60 + "\n\n")

        file.write("SHAPE\n")
        file.write(str(df.shape))
        file.write("\n\n")

        file.write("INFO\n")
        from io import StringIO
        buffer = StringIO()
        df.info(buf=buffer)
        file.write(buffer.getvalue())

        file.write("\n\nDESCRIBE\n")
        file.write(str(df.describe(include="all")))

        file.write("\n\nMISSING PERCENTAGE\n")
        missing_percentage = df.isnull().mean() * 100
        file.write(str(missing_percentage))

    print("Profile saved:", profile_file)

    # Print missing percentage
    print("\nMissing percentage:")
    print((df.isnull().mean() * 100).round(2))

    # --------------------------------------------------------
    # MISSING VALUE STRATEGY
    # --------------------------------------------------------

    missing_before = (df.isnull().mean() * 100).round(2)

    # Age missing percentage is between 5% and 30%
    # Therefore median imputation is used.
    if 5 <= missing_before["age"] <= 30:
        age_median = df["age"].median()
        df["age"] = df["age"].fillna(age_median)
        print("\nAge missing values:")
        print("Median imputation used.")
        print("Age median:", age_median)

    # Embarked has less than 5% missing.
    # Drop affected rows.
    if missing_before["embarked"] < 5:
        df = df.dropna(subset=["embarked"])
        print("Rows with missing embarked values dropped.")

    # embark_town is redundant with embarked.
    # If missing, drop affected rows.
    if "embark_town" in df.columns:
        if missing_before["embark_town"] < 5:
            df = df.dropna(subset=["embark_town"])
            print("Rows with missing embark_town values dropped.")

    # Deck has very high missing percentage.
    # Drop the column because it has too much missing data.
    if missing_before["deck"] > 30:
        df = df.drop(columns=["deck"])
        print(
            f"Deck column dropped because missing percentage "
            f"was {missing_before['deck']:.2f}%."
        )

    # Save cleaned data
    cleaned_csv = os.path.join(BASE_DIR, "titanic_cleaned.csv")
    df.to_csv(cleaned_csv, index=False)

    print("\nCleaned shape:", df.shape)
    print("\nRemaining missing values:")
    print(df.isnull().sum())

    return df


# ============================================================
# STEP 2: UNIVARIATE ANALYSIS
# ============================================================

def univariate_analysis(df):

    print("\n" + "=" * 70)
    print("STEP 2: UNIVARIATE ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # AGE HISTOGRAM
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))
    plt.hist(df["age"], bins=20)
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.title("Age Histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "age_histogram.png"))
    plt.close()

    # --------------------------------------------------------
    # AGE BOXPLOT
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))
    plt.boxplot(df["age"])
    plt.ylabel("Age")
    plt.title("Age Boxplot")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "age_boxplot.png"))
    plt.close()

    # --------------------------------------------------------
    # FARE HISTOGRAM
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))
    plt.hist(df["fare"], bins=30)
    plt.xlabel("Fare")
    plt.ylabel("Frequency")
    plt.title("Fare Histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "fare_histogram.png"))
    plt.close()

    # --------------------------------------------------------
    # FARE BOXPLOT
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))
    plt.boxplot(df["fare"])
    plt.ylabel("Fare")
    plt.title("Fare Boxplot")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "fare_boxplot.png"))
    plt.close()

    # --------------------------------------------------------
    # IQR OUTLIERS
    # --------------------------------------------------------

    def count_iqr_outliers(series):

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = ((series < lower) | (series > upper)).sum()

        return count, lower, upper

    age_outliers, age_lower, age_upper = count_iqr_outliers(df["age"])
    fare_outliers, fare_lower, fare_upper = count_iqr_outliers(df["fare"])

    print("\nIQR outlier counts:")
    print("Age outliers:", age_outliers)
    print("Fare outliers:", fare_outliers)

    # --------------------------------------------------------
    # FARE STATISTICS
    # --------------------------------------------------------

    fare_mean = df["fare"].mean()
    fare_median = df["fare"].median()
    fare_mode = df["fare"].mode().iloc[0]
    fare_skew = df["fare"].skew()

    print("\nFare statistics:")
    print("Mean:", fare_mean)
    print("Median:", fare_median)
    print("Mode:", fare_mode)
    print("Skew:", fare_skew)

    if fare_mean > fare_median > fare_mode:
        skew_direction = "Right-skewed"
        ordering = "Mean > Median > Mode"
    elif fare_mean < fare_median < fare_mode:
        skew_direction = "Left-skewed"
        ordering = "Mean < Median < Mode"
    else:
        skew_direction = "Approximately symmetric or mixed"
        ordering = "Mean, median and mode do not follow a simple ordering"

    print("Fare skew direction:", skew_direction)
    print("Ordering:", ordering)

    with open(
        os.path.join(BASE_DIR, "univariate_summary.txt"),
        "w",
        encoding="utf-8"
    ) as file:

        file.write("UNIVARIATE ANALYSIS\n")
        file.write("=" * 50 + "\n\n")

        file.write(f"Age IQR outliers: {age_outliers}\n")
        file.write(f"Fare IQR outliers: {fare_outliers}\n\n")

        file.write(f"Fare mean: {fare_mean}\n")
        file.write(f"Fare median: {fare_median}\n")
        file.write(f"Fare mode: {fare_mode}\n")
        file.write(f"Fare skew: {fare_skew}\n")
        file.write(f"Skew direction: {skew_direction}\n")
        file.write(f"Ordering: {ordering}\n")


# ============================================================
# STEP 3: BIVARIATE ANALYSIS
# ============================================================

def bivariate_analysis(df):

    print("\n" + "=" * 70)
    print("STEP 3: BIVARIATE ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # SURVIVAL RATE BY SEX USING BOOLEAN MASKS
    # --------------------------------------------------------

    female_rate = df.loc[df["sex"] == "female", "survived"].mean()
    male_rate = df.loc[df["sex"] == "male", "survived"].mean()

    print("\nSurvival rate by sex:")
    print("Female:", female_rate)
    print("Male:", male_rate)

    # --------------------------------------------------------
    # SURVIVAL RATE BY PCLASS USING BOOLEAN MASKS
    # --------------------------------------------------------

    pclass_rates = {}

    for pclass in sorted(df["pclass"].unique()):
        rate = df.loc[df["pclass"] == pclass, "survived"].mean()
        pclass_rates[pclass] = rate

    print("\nSurvival rate by pclass:")
    for pclass, rate in pclass_rates.items():
        print(f"Class {pclass}: {rate}")

    # --------------------------------------------------------
    # SEX + PCLASS
    # --------------------------------------------------------

    print("\nSurvival rate by sex + pclass:")

    sex_pclass_results = []

    for sex in ["male", "female"]:
        for pclass in sorted(df["pclass"].unique()):

            mask = (df["sex"] == sex) & (df["pclass"] == pclass)

            rate = df.loc[mask, "survived"].mean()

            print(
                f"{sex}, class {pclass}: {rate}"
            )

            sex_pclass_results.append(
                {
                    "sex": sex,
                    "pclass": pclass,
                    "survival_rate": rate
                }
            )

    pd.DataFrame(sex_pclass_results).to_csv(
        os.path.join(BASE_DIR, "survival_rates.csv"),
        index=False
    )

    # --------------------------------------------------------
    # EXACT SIX COLUMN CORRELATION MATRIX
    # --------------------------------------------------------

    correlation_columns = [
        "survived",
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare"
    ]

    correlation_matrix = df[correlation_columns].corr()

    print("\nCorrelation matrix:")
    print(correlation_matrix)

    correlation_matrix.to_csv(
        os.path.join(BASE_DIR, "correlation_matrix.csv")
    )

    # Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm"
    )
    plt.title("Titanic Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(
        os.path.join(BASE_DIR, "correlation_heatmap.png")
    )
    plt.close()

    # --------------------------------------------------------
    # TOP TWO ABSOLUTE OFF-DIAGONAL CORRELATIONS
    # --------------------------------------------------------

    pairs = []

    for i in range(len(correlation_columns)):
        for j in range(i + 1, len(correlation_columns)):

            col1 = correlation_columns[i]
            col2 = correlation_columns[j]

            value = correlation_matrix.loc[col1, col2]

            pairs.append(
                {
                    "feature_1": col1,
                    "feature_2": col2,
                    "correlation": value,
                    "absolute_correlation": abs(value)
                }
            )

    pairs_df = pd.DataFrame(pairs)

    top_two = pairs_df.sort_values(
        "absolute_correlation",
        ascending=False
    ).head(2)

    print("\nTop two absolute off-diagonal correlations:")
    print(top_two)

    top_two.to_csv(
        os.path.join(BASE_DIR, "top_two_correlations.csv"),
        index=False
    )


# ============================================================
# STEP 4: MULTIVARIATE ANALYSIS
# ============================================================

def multivariate_analysis(df):

    print("\n" + "=" * 70)
    print("STEP 4: MULTIVARIATE ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # CHART 1
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=df,
        x="pclass",
        y="survived",
        hue="sex"
    )

    plt.title("Survival Rate by Sex and Passenger Class")
    plt.tight_layout()

    plt.savefig(
        os.path.join(BASE_DIR, "survival_sex_pclass.png")
    )

    plt.close()

    # --------------------------------------------------------
    # CHART 2
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="survived",
        y="age",
        hue="sex"
    )

    plt.title("Age Distribution by Survival and Sex")
    plt.tight_layout()

    plt.savefig(
        os.path.join(BASE_DIR, "age_survival_sex.png")
    )

    plt.close()

    # --------------------------------------------------------
    # CHART 3
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="pclass",
        y="fare",
        hue="survived"
    )

    plt.title("Fare Distribution by Class and Survival")
    plt.tight_layout()

    plt.savefig(
        os.path.join(BASE_DIR, "fare_pclass_survival.png")
    )

    plt.close()

    # --------------------------------------------------------
    # FAMILY SIZE
    # --------------------------------------------------------

    df_plot = df.copy()

    df_plot["family_size"] = (
        df_plot["sibsp"] +
        df_plot["parch"] +
        1
    )

    # --------------------------------------------------------
    # CHART 4
    # --------------------------------------------------------

    family_survival = (
        df_plot
        .groupby(["family_size", "pclass"])["survived"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(9, 5))

    sns.lineplot(
        data=family_survival,
        x="family_size",
        y="survived",
        hue="pclass",
        marker="o"
    )

    plt.title("Survival Rate by Family Size and Passenger Class")
    plt.xlabel("Family Size")
    plt.ylabel("Survival Rate")
    plt.tight_layout()

    plt.savefig(
        os.path.join(BASE_DIR, "family_size_survival_class.png")
    )

    plt.close()

    # --------------------------------------------------------
    # INTERPRETATIONS
    # --------------------------------------------------------

    interpretations = """

1. Survival Rate by Sex and Passenger Class:
Female passengers generally show higher survival rates than male passengers.
Survival also varies across passenger classes, showing that both sex and class
are related to the survival outcome.

2. Age Distribution by Survival and Sex:
The age distributions differ between survivors and non-survivors.
The chart also shows how the age pattern varies between male and female
passengers.

3. Fare Distribution by Class and Survival:
Passenger class is associated with fare levels because higher classes
generally paid higher fares. The distribution also shows differences in
fare values between survivors and non-survivors.

4. Family Size, Class and Survival:
Survival rates vary with family size and passenger class.
Very small and very large family groups can show different survival
patterns across passenger classes.
"""

    with open(
        os.path.join(BASE_DIR, "multivariate_interpretations.txt"),
        "w",
        encoding="utf-8"
    ) as file:

        file.write(interpretations)


# ============================================================
# STEP 5: Z-SCORE STANDARDIZATION
# ============================================================

def z_score_standardization(df):

    print("\n" + "=" * 70)
    print("STEP 5: Z-SCORE STANDARDIZATION")
    print("=" * 70)

    result = df.copy()

    print("\nBefore standardization:")

    before_mean_age = result["age"].mean()
    before_mean_fare = result["fare"].mean()

    before_std_age = result["age"].std()
    before_std_fare = result["fare"].std()

    print("Age mean:", before_mean_age)
    print("Fare mean:", before_mean_fare)
    print("Age std:", before_std_age)
    print("Fare std:", before_std_fare)

    scaler = StandardScaler()

    result[["age_z", "fare_z"]] = scaler.fit_transform(
        result[["age", "fare"]]
    )

    print("\nAfter standardization:")

    after_mean_age = result["age_z"].mean()
    after_mean_fare = result["fare_z"].mean()

    after_std_age = result["age_z"].std()
    after_std_fare = result["fare_z"].std()

    print("Age mean:", after_mean_age)
    print("Fare mean:", after_mean_fare)
    print("Age std:", after_std_age)
    print("Fare std:", after_std_fare)

    result.to_csv(
        os.path.join(BASE_DIR, "titanic_zscore.csv"),
        index=False
    )

    summary = pd.DataFrame(
        {
            "feature": ["age", "fare"],
            "before_mean": [
                before_mean_age,
                before_mean_fare
            ],
            "before_std": [
                before_std_age,
                before_std_fare
            ],
            "after_mean": [
                after_mean_age,
                after_mean_fare
            ],
            "after_std": [
                after_std_age,
                after_std_fare
            ]
        }
    )

    summary.to_csv(
        os.path.join(BASE_DIR, "zscore_summary.csv"),
        index=False
    )


# ============================================================
# COMMON CLASSIFICATION PREPROCESSOR
# ============================================================

def create_classification_preprocessor():

    numeric_features = [
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare"
    ]

    categorical_features = [
        "sex",
        "embarked"
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


# ============================================================
# STEP 6: CLASSIFICATION MODELING
# ============================================================

def classification_modeling(df):

    print("\n" + "=" * 70)
    print("STEP 6: CLASSIFICATION MODELING")
    print("=" * 70)

    features = [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked"
    ]

    X = df[features]
    y = df["survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))

    print("\nTraining target distribution:")
    print(y_train.value_counts(normalize=True))

    print("\nTesting target distribution:")
    print(y_test.value_counts(normalize=True))

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=3,
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE
        )
    }

    results = []

    fitted_pipelines = {}

    # --------------------------------------------------------
    # TRAIN ALL THREE MODELS
    # --------------------------------------------------------

    for model_name, model in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    create_classification_preprocessor()
                ),
                (
                    "model",
                    model
                )
            ]
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        probabilities = pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

        print("\n" + model_name)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1:", f1)
        print("ROC-AUC:", roc_auc)

        results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "ROC_AUC": roc_auc
            }
        )

        fitted_pipelines[model_name] = pipeline

        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        cm = confusion_matrix(
            y_test,
            predictions
        )

        plt.figure(figsize=(5, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues"
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(
            f"Confusion Matrix - {model_name}"
        )

        filename = (
            model_name
            .lower()
            .replace(" ", "_")
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                BASE_DIR,
                f"confusion_matrix_{filename}.png"
            )
        )

        plt.close()

    # --------------------------------------------------------
    # COMPARISON TABLE
    # --------------------------------------------------------

    comparison = pd.DataFrame(results)

    print("\nClassification comparison:")
    print(comparison.to_string(index=False))

    comparison.to_csv(
        os.path.join(
            BASE_DIR,
            "classification_comparison.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # ROC CURVES
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    for model_name, pipeline in fitted_pipelines.items():

        probabilities = pipeline.predict_proba(
            X_test
        )[:, 1]

        fpr, tpr, _ = roc_curve(
            y_test,
            probabilities
        )

        auc_value = roc_auc_score(
            y_test,
            probabilities
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{model_name} AUC={auc_value:.3f}"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(BASE_DIR, "roc_curves.png")
    )

    plt.close()

    # --------------------------------------------------------
    # DECISION TREE PLOT
    # --------------------------------------------------------

    tree_pipeline = fitted_pipelines["Decision Tree"]

    tree_model = tree_pipeline.named_steps["model"]

    preprocessor = tree_pipeline.named_steps["preprocessor"]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    plt.figure(figsize=(20, 10))

    plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=[
            "Not Survived",
            "Survived"
        ],
        filled=True,
        rounded=True
    )

    plt.title("Decision Tree")
    plt.tight_layout()

    plt.savefig(
        os.path.join(BASE_DIR, "decision_tree.png")
    )

    plt.close()

    return X_train, X_test, y_train, y_test, fitted_pipelines, comparison


# ============================================================
# STEP 7: CLASS IMBALANCE
# BASELINE vs BALANCED vs SMOTE
# ============================================================

def imbalance_analysis(df):

    print("\n" + "=" * 70)
    print("STEP 7: CLASS IMBALANCE ANALYSIS")
    print("=" * 70)

    features = [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked"
    ]

    X = df[features]
    y = df["survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    results = []

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    baseline_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_classification_preprocessor()
            ),
            (
                "model",
                LogisticRegression(max_iter=1000)
            )
        ]
    )

    baseline_pipeline.fit(
        X_train,
        y_train
    )

    baseline_predictions = baseline_pipeline.predict(
        X_test
    )

    # --------------------------------------------------------
    # BALANCED CLASS WEIGHT
    # --------------------------------------------------------

    balanced_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_classification_preprocessor()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced"
                )
            )
        ]
    )

    balanced_pipeline.fit(
        X_train,
        y_train
    )

    balanced_predictions = balanced_pipeline.predict(
        X_test
    )

    # --------------------------------------------------------
    # SMOTE
    # --------------------------------------------------------

    # Dense encoder is used because SMOTE works directly
    # with the transformed training matrix.
    numeric_features = [
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare"
    ]

    categorical_features = [
        "sex",
        "embarked"
    ]

    smote_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            )
                        ),
                        (
                            "scaler",
                            StandardScaler()
                        )
                    ]
                ),
                numeric_features
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            )
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False
                            )
                        )
                    ]
                ),
                categorical_features
            )
        ]
    )

    # IMPORTANT:
    # Fit preprocessing ONLY on training data.
    X_train_processed = smote_preprocessor.fit_transform(
        X_train
    )

    X_test_processed = smote_preprocessor.transform(
        X_test
    )

    print("\nClass distribution before SMOTE:")
    print(y_train.value_counts())

    smote = SMOTE(
        random_state=RANDOM_STATE
    )

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train_processed,
        y_train
    )

    print("\nClass distribution after SMOTE:")
    print(pd.Series(y_train_smote).value_counts())

    smote_model = LogisticRegression(
        max_iter=1000
    )

    smote_model.fit(
        X_train_smote,
        y_train_smote
    )

    smote_predictions = smote_model.predict(
        X_test_processed
    )

    # --------------------------------------------------------
    # FUNCTION TO SAVE METRICS
    # --------------------------------------------------------

    def calculate_metrics(
        method_name,
        y_true,
        predictions
    ):

        return {
            "Method": method_name,
            "Precision": precision_score(
                y_true,
                predictions,
                zero_division=0
            ),
            "Recall": recall_score(
                y_true,
                predictions,
                zero_division=0
            ),
            "F1": f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        }

    results.append(
        calculate_metrics(
            "Baseline",
            y_test,
            baseline_predictions
        )
    )

    results.append(
        calculate_metrics(
            "class_weight='balanced'",
            y_test,
            balanced_predictions
        )
    )

    results.append(
        calculate_metrics(
            "SMOTE",
            y_test,
            smote_predictions
        )
    )

    imbalance_df = pd.DataFrame(results)

    print("\nImbalance comparison:")
    print(imbalance_df.to_string(index=False))

    imbalance_df.to_csv(
        os.path.join(
            BASE_DIR,
            "imbalance_comparison.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # IMBALANCE BAR CHART
    # --------------------------------------------------------

    imbalance_melted = imbalance_df.melt(
        id_vars="Method",
        value_vars=[
            "Precision",
            "Recall",
            "F1"
        ],
        var_name="Metric",
        value_name="Score"
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=imbalance_melted,
        x="Method",
        y="Score",
        hue="Metric"
    )

    plt.ylim(0, 1)
    plt.title("Class Imbalance Comparison")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            BASE_DIR,
            "imbalance_comparison.png"
        )
    )

    plt.close()

    # --------------------------------------------------------
    # SHORT CONCLUSION
    # --------------------------------------------------------

    highest_recall = imbalance_df.loc[
        imbalance_df["Recall"].idxmax()
    ]

    highest_f1 = imbalance_df.loc[
        imbalance_df["F1"].idxmax()
    ]

    conclusion = f"""
Class imbalance analysis conclusion:

The method with the highest recall in this test set was
{highest_recall['Method']} with recall
{highest_recall['Recall']:.4f}.

The method with the highest F1 score was
{highest_f1['Method']} with F1 score
{highest_f1['F1']:.4f}.

SMOTE was applied only to the training fold after the preprocessing
step was fitted on training data. The test data was kept unchanged
for evaluation.
"""

    print(conclusion)

    with open(
        os.path.join(
            BASE_DIR,
            "imbalance_conclusion.txt"
        ),
        "w",
        encoding="utf-8"
    ) as file:

        file.write(conclusion)


# ============================================================
# STEP 8: RANDOM FOREST GRIDSEARCH + OOB
# ============================================================

def random_forest_tuning(df):

    print("\n" + "=" * 70)
    print("STEP 8: RANDOM FOREST GRIDSEARCH + OOB")
    print("=" * 70)

    features = [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked"
    ]

    X = df[features]
    y = df["survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    rf_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_classification_preprocessor()
            ),
            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    oob_score=True,
                    bootstrap=True
                )
            )
        ]
    )

    param_grid = {
        "model__n_estimators": [
            100,
            200
        ],

        "model__max_depth": [
            3,
            5,
            None
        ],

        "model__max_features": [
            "sqrt",
            "log2"
        ]
    }

    grid_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    grid_search.fit(
        X_train,
        y_train
    )

    best_model = grid_search.best_estimator_

    predictions = best_model.predict(
        X_test
    )

    best_params = grid_search.best_params_

    best_cv_score = grid_search.best_score_

    oob_score = (
        best_model
        .named_steps["model"]
        .oob_score_
    )

    print("\nBest parameters:")
    print(best_params)

    print("\nBest CV F1 score:")
    print(best_cv_score)

    print("\nOOB score:")
    print(oob_score)

    with open(
        os.path.join(
            BASE_DIR,
            "random_forest_tuning.txt"
        ),
        "w",
        encoding="utf-8"
    ) as file:

        file.write("RANDOM FOREST GRIDSEARCH\n")
        file.write("=" * 50 + "\n\n")

        file.write(
            f"Best parameters:\n{best_params}\n\n"
        )

        file.write(
            f"Best CV F1 score: {best_cv_score}\n"
        )

        file.write(
            f"OOB score: {oob_score}\n"
        )

    # Save best RF
    joblib.dump(
        best_model,
        os.path.join(
            BASE_DIR,
            "best_random_forest_pipeline.joblib"
        )
    )

    return best_model


# ============================================================
# STEP 9: MULTIVARIATE LINEAR REGRESSION
# Predict FARE
# ============================================================

def regression_analysis(df):

    print("\n" + "=" * 70)
    print("STEP 9: MULTIVARIATE LINEAR REGRESSION")
    print("=" * 70)

    features = [
        "pclass",
        "age",
        "sibsp",
        "parch",
        "sex",
        "embarked"
    ]

    X = df[features]
    y = df["fare"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    numeric_features = [
        "pclass",
        "age",
        "sibsp",
        "parch"
    ]

    categorical_features = [
        "sex",
        "embarked"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            )
                        ),
                        (
                            "scaler",
                            StandardScaler()
                        )
                    ]
                ),
                numeric_features
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            )
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore"
                            )
                        )
                    ]
                ),
                categorical_features
            )
        ]
    )

    regression_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                LinearRegression()
            )
        ]
    )

    regression_pipeline.fit(
        X_train,
        y_train
    )

    predictions = regression_pipeline.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    n = len(y_test)

    transformed_X_test = (
        regression_pipeline
        .named_steps["preprocessor"]
        .transform(X_test)
    )

    p = transformed_X_test.shape[1]

    if n - p - 1 > 0:
        adjusted_r2 = (
            1
            - (
                (1 - r2) * (n - 1)
                / (n - p - 1)
            )
        )
    else:
        adjusted_r2 = np.nan

    residuals = y_test - predictions

    print("\nRegression metrics:")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)
    print("Adjusted R2:", adjusted_r2)

    # --------------------------------------------------------
    # RESIDUAL PLOT
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.scatter(
        predictions,
        residuals,
        alpha=0.6
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xlabel("Predicted Fare")
    plt.ylabel("Residual")
    plt.title("Residual Plot")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            BASE_DIR,
            "regression_residual_plot.png"
        )
    )

    plt.close()

    # --------------------------------------------------------
    # HETEROSCEDASTICITY CHECK
    # --------------------------------------------------------

    residual_check = pd.DataFrame(
        {
            "predicted": predictions,
            "residual": residuals
        }
    )

    residual_check["abs_residual"] = (
        residual_check["residual"].abs()
    )

    if len(residual_check) >= 10:

        residual_check["prediction_group"] = pd.qcut(
            residual_check["predicted"],
            q=4,
            duplicates="drop"
        )

        group_std = (
            residual_check
            .groupby(
                "prediction_group",
                observed=True
            )["residual"]
            .std()
        )

        if len(group_std) >= 2:

            min_std = group_std.min()
            max_std = group_std.max()

            if min_std > 0:
                ratio = max_std / min_std
            else:
                ratio = np.inf

            if ratio > 2:
                hetero_statement = (
                    "Residual spread varies substantially across "
                    "prediction ranges, suggesting possible "
                    "heteroscedasticity."
                )
            else:
                hetero_statement = (
                    "Residual spread is relatively similar across "
                    "prediction ranges, so strong heteroscedasticity "
                    "is not evident from this check."
                )

        else:
            hetero_statement = (
                "There were not enough prediction groups for a "
                "heteroscedasticity check."
            )

    else:
        hetero_statement = (
            "There were not enough test observations for a "
            "heteroscedasticity check."
        )

    print("\nHeteroscedasticity statement:")
    print(hetero_statement)

    with open(
        os.path.join(
            BASE_DIR,
            "regression_summary.txt"
        ),
        "w",
        encoding="utf-8"
    ) as file:

        file.write("MULTIVARIATE LINEAR REGRESSION\n")
        file.write("=" * 50 + "\n\n")

        file.write(f"MAE: {mae}\n")
        file.write(f"RMSE: {rmse}\n")
        file.write(f"R2: {r2}\n")
        file.write(f"Adjusted R2: {adjusted_r2}\n\n")

        file.write(
            "Heteroscedasticity statement:\n"
        )

        file.write(
            hetero_statement
        )

    regression_metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted_R2": adjusted_r2
    }

    pd.DataFrame(
        [regression_metrics]
    ).to_csv(
        os.path.join(
            BASE_DIR,
            "regression_metrics.csv"
        ),
        index=False
    )

    return regression_metrics


# ============================================================
# STEP 10: FINAL COMPARISON + CLASSIFIER RECOMMENDATION
# ============================================================

def final_comparison(classification_df, regression_metrics):

    print("\n" + "=" * 70)
    print("STEP 10: FINAL COMPARISON")
    print("=" * 70)

    # Classification table
    classification_final = classification_df.copy()

    classification_final["Task"] = "Classification"

    classification_final = classification_final[
        [
            "Task",
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC"
        ]
    ]

    # Regression table
    regression_final = pd.DataFrame(
        [
            {
                "Task": "Regression",
                "Model": "Linear Regression",
                "MAE": regression_metrics["MAE"],
                "RMSE": regression_metrics["RMSE"],
                "R2": regression_metrics["R2"],
                "Adjusted_R2":
                    regression_metrics["Adjusted_R2"]
            }
        ]
    )

    print("\nClassification metrics:")
    print(classification_final.to_string(index=False))

    print("\nRegression metrics:")
    print(regression_final.to_string(index=False))

    classification_final.to_csv(
        os.path.join(
            BASE_DIR,
            "final_classification_metrics.csv"
        ),
        index=False
    )

    regression_final.to_csv(
        os.path.join(
            BASE_DIR,
            "final_regression_metrics.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # CLASSIFIER RECOMMENDATION BASED ON F1
    # --------------------------------------------------------

    selected_row = classification_df.loc[
        classification_df["F1"].idxmax()
    ]

    selected_model = selected_row["Model"]

    recommendation = f"""
Final classifier recommendation:

The classification models were compared using accuracy,
precision, recall, F1 score and ROC-AUC.

Based on the F1 score on the fixed test set, the model selected
for the final pipeline is {selected_model}.

The final model should still be interpreted using all reported
metrics rather than F1 alone because precision, recall and ROC-AUC
describe different aspects of classification performance.

The complete fitted pipeline for the selected classifier is saved
and will be reloaded in the next step for a raw-input prediction.
"""

    print(recommendation)

    with open(
        os.path.join(
            BASE_DIR,
            "final_recommendation.txt"
        ),
        "w",
        encoding="utf-8"
    ) as file:

        file.write(recommendation)

    return selected_model


# ============================================================
# STEP 11: SAVE + RELOAD COMPLETE FITTED PIPELINE
# ============================================================

def save_and_reload_pipeline(
    df,
    selected_model
):

    print("\n" + "=" * 70)
    print("STEP 11: SAVE AND RELOAD FITTED PIPELINE")
    print("=" * 70)

    features = [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked"
    ]

    X = df[features]
    y = df["survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    if selected_model == "Logistic Regression":

        model = LogisticRegression(
            max_iter=1000
        )

    elif selected_model == "Decision Tree":

        model = DecisionTreeClassifier(
            max_depth=3,
            random_state=RANDOM_STATE
        )

    else:

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE
        )

    final_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_classification_preprocessor()
            ),
            (
                "model",
                model
            )
        ]
    )

    final_pipeline.fit(
        X_train,
        y_train
    )

    pipeline_path = os.path.join(
        BASE_DIR,
        "final_classifier_pipeline.joblib"
    )

    joblib.dump(
        final_pipeline,
        pipeline_path
    )

    print("\nPipeline saved:")
    print(pipeline_path)

    # --------------------------------------------------------
    # RELOAD
    # --------------------------------------------------------

    loaded_pipeline = joblib.load(
        pipeline_path
    )

    # --------------------------------------------------------
    # RAW INPUT EXAMPLE
    # --------------------------------------------------------

    raw_input = pd.DataFrame(
        [
            {
                "pclass": 3,
                "sex": "female",
                "age": 25,
                "sibsp": 0,
                "parch": 0,
                "fare": 15.0,
                "embarked": "S"
            }
        ]
    )

    prediction = loaded_pipeline.predict(
        raw_input
    )

    probability = loaded_pipeline.predict_proba(
        raw_input
    )[:, 1]

    print("\nRaw input:")
    print(raw_input)

    print("\nReloaded pipeline prediction:")

    if prediction[0] == 1:
        print("Survived")
    else:
        print("Not Survived")

    print("Survival probability:", probability[0])

    with open(
        os.path.join(
            BASE_DIR,
            "pipeline_prediction.txt"
        ),
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            f"Selected model: {selected_model}\n"
        )

        file.write(
            f"Prediction: {prediction[0]}\n"
        )

        file.write(
            f"Survival probability: {probability[0]}\n"
        )


# ============================================================
# MAIN PROGRAM
# ONE COMMAND RUNS EVERYTHING
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("ZEpto AI CAPSTONE - MODULE 2")
    print("TITANIC ANALYTICS + MACHINE LEARNING")
    print("=" * 70)

    # Step 1
    df = load_and_clean_data()

    # Step 2
    univariate_analysis(df)

    # Step 3
    bivariate_analysis(df)

    # Step 4
    multivariate_analysis(df)

    # Step 5
    z_score_standardization(df)

    # Step 6
    (
        X_train,
        X_test,
        y_train,
        y_test,
        fitted_pipelines,
        classification_df
    ) = classification_modeling(df)

    # Step 7
    imbalance_analysis(df)

    # Step 8
    best_rf = random_forest_tuning(df)

    # Step 9
    regression_metrics = regression_analysis(df)

    # Step 10
    selected_model = final_comparison(
        classification_df,
        regression_metrics
    )

    # Step 11
    save_and_reload_pipeline(
        df,
        selected_model
    )

    print("\n")
    print("=" * 70)
    print("MODULE 2 COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    print("\nAll generated files are inside:")
    print(BASE_DIR)
