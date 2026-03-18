# Used Car Price Prediction

An end-to-end machine learning web app that predicts a used car resale price from vehicle details.

## Project Structure

```text
Used Car Price Prediction/
|- data/
|  |- used_cars.csv
|- model/
|  |- train_model.py
|- backend/
|  |- main.py
|  |- car_price_model.pkl   # generated after training
|- frontend/
|  |- src/
|  |- package.json
|- requirements.txt
```

## Trained Model Workflow (Detailed)

The model training logic is implemented in `model/train_model.py`.

### 1) Load and Clean Data

- Reads dataset from `data/used_cars.csv`.
- Cleans `price` by removing currency symbols and commas, then converts to numeric.
- Drops rows where target price is missing.

### 2) Select Input Columns

Raw inputs used to build model features:

- `brand`
- `model_year`
- `milage`
- `fuel_type`
- `engine`
- `transmission`
- `accident`
- `clean_title`

### 3) Feature Engineering

`build_features()` transforms raw columns into model-ready features:

- Numeric features:
  - `milage`
  - `horsepower` (parsed from engine text, for example `300HP`)
  - `engine_size` (parsed from text, for example `3.5L`)
  - `cylinders` (parsed from terms like `V6`, `4 Cylinder`)
  - `car_age` (`REFERENCE_YEAR - model_year`, reference year is `2026`)
  - `had_accident` (binary flag from accident text)
- Categorical features:
  - `fuel_type`
  - `clean_title`
  - `brand_category` (`Luxury` / `Premium` / `Economy`)
  - `transmission_type` (`Manual` / `CVT` / `Automatic`)

### 4) Target Transformation

- Uses `y = log1p(price)` to reduce price skew and stabilize regression training.

### 5) Train / Validation / Test Split

- 80% train+validation, 20% test.
- Then train+validation is split again: 80% train, 20% validation.

### 6) Outlier Removal (Training Only)

- IQR-based filtering on:
  - feature columns: `milage`, `horsepower`, `engine_size`
  - target price (log-scaled target series)
- Done only on training data to reduce distortion from extreme points.

### 7) Preprocessing Pipeline

`build_pipeline()` creates a sklearn pipeline:

- Numeric transformer:
  - `SimpleImputer(strategy="median")`
- Categorical transformer:
  - `SimpleImputer(strategy="most_frequent")`
  - `OneHotEncoder(handle_unknown="ignore")`
- Final estimator appended as `model`.

### 8) Model Candidates and Selection

Three regressors are evaluated on validation data:

- `LinearRegression`
- `RandomForestRegressor`
- `GradientBoostingRegressor`

Selection criterion:

- Sort by lowest validation MAE (after converting predictions back from log scale).

### 9) Final Training and Test Evaluation

- Rebuilds training data using both train + validation sets.
- Reapplies outlier removal.
- Trains a new pipeline with the selected best estimator.
- Evaluates on held-out test set with:
  - MAE
  - RMSE
  - R2

### 10) Model Artifact Export

Saves to:

- `backend/car_price_model.pkl`

Saved object format:

- `pipeline`: trained sklearn pipeline
- `reference_year`: `2026`
- `features_version`: `1`

## Whole Project Workflow

### A) Training Phase

1. Run `model/train_model.py`.
2. Script trains and selects best model.
3. Model artifact is written to `backend/car_price_model.pkl`.

### B) Backend Serving Phase (FastAPI)

File: `backend/main.py`

1. On request, backend loads model artifact (cached with `lru_cache`).
2. Validates request payload using `CarInput` schema.
3. Builds engineered features using the same transformation logic as training.
4. Predicts log-price using trained pipeline.
5. Converts prediction back to normal price with `expm1`.
6. Returns response:

```json
{ "predicted_price": 23456.78 }
```

Available endpoints:

- `GET /` basic status message
- `GET /health` health + model availability check
- `POST /predict` price prediction endpoint

### C) Frontend Phase (React + Vite)

Key files:

- `frontend/src/components/CarForm.jsx`
- `frontend/src/App.jsx`

Workflow:

1. User enters 8 car details in form UI.
2. Frontend performs basic validation.
3. Sends POST request to backend `/predict`.
4. Displays formatted predicted price.

Backend URL config:

- Uses `VITE_API_BASE_URL` env variable if present.
- Defaults to `http://127.0.0.1:8000`.

## How To Run

## 1) Install Python Dependencies

```bash
pip install -r requirements.txt
```

## 2) Train the Model

```bash
python model/train_model.py
```

This creates `backend/car_price_model.pkl`.

## 3) Run Backend API

```bash
uvicorn backend.main:app --reload
```

Backend runs at `http://127.0.0.1:8000`.

## 4) Run Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://127.0.0.1:5173`.

## Tech Stack

- Python, pandas, numpy, scikit-learn
- FastAPI + Uvicorn
- React + Vite
- Axios
