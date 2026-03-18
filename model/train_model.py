from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "used_cars.csv"
MODEL_OUTPUTS = [
    PROJECT_ROOT / "backend" / "car_price_model.pkl",
]

REFERENCE_YEAR = 2026

NUMERIC_FEATURES = [
    "milage",
    "horsepower",
    "engine_size",
    "cylinders",
    "car_age",
    "had_accident",
]

CATEGORICAL_FEATURES = [
    "fuel_type",
    "clean_title",
    "brand_category",
    "transmission_type",
]


def clean_currency(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .replace({"nan": np.nan, "None": np.nan})
        .astype(float)
    )


def clean_milage(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(" mi.", "", regex=False)
        .str.replace(",", "", regex=False)
        .replace({"nan": np.nan, "None": np.nan})
        .astype(float)
    )


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def extract_first_float(text: str, patterns: Iterable[str]) -> float | np.floating:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            for group in match.groups():
                if group is not None:
                    return float(group)
    return np.nan


def parse_horsepower(engine: str) -> float | np.floating:
    return extract_first_float(engine, [r"(\d+\.?\d*)\s*HP"])


def parse_engine_size(engine: str) -> float | np.floating:
    return extract_first_float(engine, [r"(\d+\.?\d*)\s*L", r"(\d+\.?\d*)\s*Liter"])


def parse_cylinders(engine: str) -> float | np.floating:
    return extract_first_float(
        engine,
        [r"(\d+)\s*Cylinder", r"\bV(\d+)\b", r"\bI(\d+)\b", r"\bH(\d+)\b"],
    )


def brand_group(brand: object) -> str:
    value = normalize_text(brand).lower()

    luxury_tokens = [
        "bmw",
        "mercedes",
        "porsche",
        "audi",
        "lexus",
        "tesla",
        "jaguar",
        "maserati",
        "bentley",
        "lamborghini",
        "ferrari",
        "rolls-royce",
        "aston",
        "mclaren",
        "bugatti",
        "maybach",
        "lotus",
        "lucid",
        "polestar",
        "karma",
        "rivian",
        "genesis",
        "alfa",
    ]

    premium_tokens = [
        "cadillac",
        "acura",
        "infiniti",
        "volvo",
        "lincoln",
        "mini",
        "buick",
        "land rover",
    ]

    if any(token in value for token in luxury_tokens):
        return "Luxury"
    if any(token in value for token in premium_tokens):
        return "Premium"
    return "Economy"


def transmission_group(transmission: object) -> str:
    value = normalize_text(transmission).lower()
    if "manual" in value:
        return "Manual"
    if "cvt" in value:
        return "CVT"
    return "Automatic"


def accident_flag(accident: object) -> int:
    value = normalize_text(accident).lower()
    return int("accident" in value or "damage" in value)


def clean_title_flag(clean_title: object) -> str:
    value = normalize_text(clean_title).lower()
    if value in {"yes", "y", "true", "1"}:
        return "Yes"
    if value in {"no", "n", "false", "0"}:
        return "No"
    return "Unknown"


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()

    if "milage" in data.columns:
        data["milage"] = clean_milage(data["milage"])
    if "model_year" in data.columns:
        data["model_year"] = pd.to_numeric(data["model_year"], errors="coerce")

    engine_text = data.get("engine", pd.Series("", index=data.index)).map(normalize_text)
    data["horsepower"] = engine_text.map(parse_horsepower)
    data["engine_size"] = engine_text.map(parse_engine_size)
    data["cylinders"] = engine_text.map(parse_cylinders)
    data["car_age"] = REFERENCE_YEAR - data["model_year"]
    data["had_accident"] = data.get("accident", pd.Series("", index=data.index)).map(accident_flag)
    data["brand_category"] = data.get("brand", pd.Series("", index=data.index)).map(brand_group)
    data["transmission_type"] = data.get("transmission", pd.Series("", index=data.index)).map(
        transmission_group
    )
    data["fuel_type"] = data.get("fuel_type", pd.Series("Unknown", index=data.index)).fillna("Unknown")
    data["clean_title"] = data.get("clean_title", pd.Series("Unknown", index=data.index)).map(
        clean_title_flag
    )

    return data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def remove_training_outliers(frame: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    engineered = build_features(frame)
    numeric = engineered[["milage", "horsepower", "engine_size"]].copy()

    mask = pd.Series(True, index=frame.index)
    for column in numeric.columns:
        values = numeric[column]
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask &= values.between(lower, upper) | values.isna()

    target_q1 = target.quantile(0.25)
    target_q3 = target.quantile(0.75)
    target_iqr = target_q3 - target_q1
    target_lower = target_q1 - 1.5 * target_iqr
    target_upper = target_q3 + 1.5 * target_iqr
    mask &= target.between(target_lower, target_upper)

    return frame.loc[mask].copy(), target.loc[mask].copy()


def build_pipeline(model) -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessing),
            ("model", model),
        ]
    )


def evaluate_predictions(y_true_log: pd.Series, y_pred_log: np.ndarray) -> dict[str, float]:
    actual_price = np.expm1(y_true_log)
    predicted_price = np.expm1(y_pred_log)

    return {
        "mae": mean_absolute_error(actual_price, predicted_price),
        "rmse": np.sqrt(mean_squared_error(actual_price, predicted_price)),
        "r2": r2_score(y_true_log, y_pred_log),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df["price"] = clean_currency(df["price"])
    df = df.dropna(subset=["price"]).copy()

    X = df[
        [
            "brand",
            "model_year",
            "milage",
            "fuel_type",
            "engine",
            "transmission",
            "accident",
            "clean_title",
        ]
    ].copy()
    y = np.log1p(df["price"])

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.2,
        random_state=42,
    )

    X_train, y_train = remove_training_outliers(X_train, y_train)

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    validation_results = []
    for name, estimator in candidates.items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(build_features(X_train), y_train)
        predictions = pipeline.predict(build_features(X_val))
        metrics = evaluate_predictions(y_val, predictions)
        validation_results.append({"model": name, **metrics})

    results = pd.DataFrame(validation_results).sort_values(by="mae", ascending=True)
    best_model_name = results.iloc[0]["model"]
    best_estimator = clone(candidates[best_model_name])

    final_X_train, final_y_train = remove_training_outliers(
        pd.concat([X_train, X_val]),
        pd.concat([y_train, y_val]),
    )

    final_pipeline = build_pipeline(best_estimator)
    final_pipeline.fit(build_features(final_X_train), final_y_train)

    test_predictions = final_pipeline.predict(build_features(X_test))
    test_metrics = evaluate_predictions(y_test, test_predictions)

    print("Validation results:")
    print(results.to_string(index=False))
    print("\nSelected model:", best_model_name)
    print("Test metrics:")
    print(
        pd.DataFrame([test_metrics])
        .rename(columns={"mae": "MAE", "rmse": "RMSE", "r2": "R2"})
        .to_string(index=False)
    )

    for output_path in MODEL_OUTPUTS:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "pipeline": final_pipeline,
            "reference_year": REFERENCE_YEAR,
            "features_version": 1,
        }
        joblib.dump(artifact, output_path)
        print(f"Saved model to {output_path}")


if __name__ == "__main__":
    main()
