from functools import lru_cache
from pathlib import Path
import os
from datetime import datetime
import re

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATHS = [
    BASE_DIR / "car_price_model.pkl",
    BASE_DIR.parent / "model" / "car_price_model.pkl",
]
DEFAULT_ORIGINS = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
]
CURRENT_YEAR = datetime.now().year

app = FastAPI(title="Used Car Price API")

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", ",".join(DEFAULT_ORIGINS)).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def extract_first_float(text: str, patterns: list[str]) -> float | np.floating:
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


def brand_group(brand: str) -> str:
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


def transmission_group(transmission: str) -> str:
    value = normalize_text(transmission).lower()
    if "manual" in value:
        return "Manual"
    if "cvt" in value:
        return "CVT"
    return "Automatic"


def accident_flag(accident: str) -> int:
    value = normalize_text(accident).lower()
    return int("accident" in value or "damage" in value)


def clean_title_flag(clean_title: str) -> str:
    value = normalize_text(clean_title).lower()
    if value in {"yes", "y", "true", "1"}:
        return "Yes"
    if value in {"no", "n", "false", "0"}:
        return "No"
    return "Unknown"


def build_features(input_frame: pd.DataFrame, reference_year: int) -> pd.DataFrame:
    data = input_frame.copy()

    return pd.DataFrame(
        [
            {
                "milage": float(row["milage"]),
                "horsepower": parse_horsepower(row["engine"]),
                "engine_size": parse_engine_size(row["engine"]),
                "cylinders": parse_cylinders(row["engine"]),
                "car_age": reference_year - int(row["model_year"]),
                "had_accident": accident_flag(row["accident"]),
                "fuel_type": normalize_text(row["fuel_type"]) or "Unknown",
                "clean_title": clean_title_flag(row["clean_title"]),
                "brand_category": brand_group(row["brand"]),
                "transmission_type": transmission_group(row["transmission"]),
            }
            for _, row in data.iterrows()
        ]
    )


@lru_cache(maxsize=1)
def load_model():
    for path in MODEL_PATHS:
        if path.exists():
            return joblib.load(path)
    searched_paths = ", ".join(str(path) for path in MODEL_PATHS)
    raise FileNotFoundError(f"Model file not found. Expected one of: {searched_paths}")


class CarInput(BaseModel):
    brand: str = Field(..., min_length=1, max_length=80)
    model_year: int = Field(..., ge=1980, le=CURRENT_YEAR + 1)
    milage: float = Field(..., ge=0)
    fuel_type: str = Field(..., min_length=1, max_length=40)
    engine: str = Field(..., min_length=1, max_length=120)
    transmission: str = Field(..., min_length=1, max_length=60)
    accident: str = Field(..., min_length=1, max_length=120)
    clean_title: str = Field(default="Unknown", min_length=1, max_length=20)

    @field_validator("brand", "fuel_type", "engine", "transmission", "accident", "clean_title")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Field cannot be empty")
        return value


@app.get("/")
def home():
    return {"message": "Used Car Price API running"}


@app.get("/health")
def health_check():
    try:
        load_model()
        return {"status": "ok"}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict")
def predict(data: CarInput):
    try:
        artifact = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if isinstance(artifact, dict):
        model = artifact["pipeline"]
        reference_year = int(artifact.get("reference_year", CURRENT_YEAR))
    else:
        model = artifact
        reference_year = CURRENT_YEAR

    input_frame = pd.DataFrame(
        [
            {
                "brand": data.brand,
                "model_year": data.model_year,
                "milage": data.milage,
                "fuel_type": data.fuel_type,
                "engine": data.engine,
                "transmission": data.transmission,
                "accident": data.accident,
                "clean_title": data.clean_title,
            }
        ]
    )

    prediction_log = model.predict(build_features(input_frame, reference_year))
    predicted_price = float(np.expm1(prediction_log)[0])

    return {"predicted_price": round(predicted_price, 2)}
