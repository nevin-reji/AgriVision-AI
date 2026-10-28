import json
import math
import sys
from pathlib import Path

if __package__ in {None, ''}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.__init__ import BASE_DIR

DATASET_PATH = BASE_DIR / 'datasets' / 'crop_yield.csv'
MODEL_PATH = BASE_DIR / 'models' / 'yield_model.joblib'


def create_demo_yield_dataset(output_path: Path | None = None) -> Path:
    output_path = output_path or DATASET_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    crop_names = ['rice', 'wheat', 'maize', 'cotton', 'sugarcane']
    states = ['Punjab', 'Maharashtra', 'Uttar Pradesh', 'Karnataka', 'Tamil Nadu']
    seasons = ['Kharif', 'Rabi', 'Zaid']
    records = []

    for year in range(2010, 2024):
        for state in states:
            for crop in crop_names:
                base_area = 800 if crop in {'rice', 'wheat'} else 600
                area = base_area + (year - 2010) * 2 + (hash((state, crop)) % 50)
                temp = 20 + (hash((state, crop)) % 15) + (year - 2010) * 0.08
                rainfall = 500 + (hash((state, crop)) % 300) + (year - 2010) * 2
                production = area * (3.2 + (hash((state, crop)) % 8) * 0.2 + (year - 2010) * 0.03)
                records.append({
                    'crop': crop,
                    'state': state,
                    'district': f'{state[:4]}-{year % 100}',
                    'year': year,
                    'season': seasons[(year + hash((state, crop)) % 3) % len(seasons)],
                    'area_ha': round(area, 2),
                    'production_tonnes': round(production, 2),
                    'rainfall_mm': round(rainfall, 2),
                    'temperature_c': round(temp, 2),
                    'soil_ph': round(6 + (hash((state, crop)) % 10) / 10, 2),
                    'soil_moisture_pct': round(38 + (hash((state, crop)) % 25), 2),
                })

    df = pd.DataFrame(records)
    df['yield_t_ha'] = df['production_tonnes'] / df['area_ha']
    df = df[['crop', 'state', 'district', 'year', 'season', 'area_ha', 'production_tonnes', 'rainfall_mm', 'temperature_c', 'soil_ph', 'soil_moisture_pct', 'yield_t_ha']]
    df.to_csv(output_path, index=False)
    return output_path


def ensure_yield_dataset(dataset_path: Path | None = None) -> Path:
    dataset_path = dataset_path or DATASET_PATH
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if not dataset_path.exists():
        print(f'Warning: dataset not found at {dataset_path}. Creating a demo yield dataset for local testing.')
        return create_demo_yield_dataset(dataset_path)
    return dataset_path


def _prepare_features(df: pd.DataFrame):
    categorical = ['crop', 'state', 'district', 'season']
    numeric = ['year', 'area_ha', 'rainfall_mm', 'temperature_c', 'soil_ph', 'soil_moisture_pct']
    return categorical, numeric


def train_yield_model(dataset_path: Path | None = None, model_path: Path | None = None):
    dataset_path = ensure_yield_dataset(dataset_path or DATASET_PATH)
    model_path = model_path or MODEL_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    required = {'crop', 'state', 'district', 'year', 'season', 'area_ha', 'rainfall_mm', 'temperature_c', 'soil_ph', 'soil_moisture_pct'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'Yield dataset missing required columns: {missing}')
    if 'yield_t_ha' not in df.columns:
        df['yield_t_ha'] = df['production_tonnes'] / df['area_ha']
    df = df[(df['area_ha'] > 0) & (df['production_tonnes'] >= 0)].copy()

    categorical, numeric = _prepare_features(df)
    X = df[categorical + numeric]
    y = df['yield_t_ha']

    train_mask = df['year'] <= df['year'].median()
    X_train = X[train_mask]
    X_test = X[~train_mask]
    y_train = y[train_mask]
    y_test = y[~train_mask]

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
        ('num', StandardScaler(), numeric),
    ])

    models = {
        'linear_regression': LinearRegression(),
        'random_forest': RandomForestRegressor(n_estimators=200, random_state=42),
    }

    results = {}
    for name, estimator in models.items():
        pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('model', estimator),
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        results[name] = {
            'mae': mean_absolute_error(y_test, preds),
            'rmse': math.sqrt(mean_squared_error(y_test, preds)),
            'r2': r2_score(y_test, preds),
            'pipeline': pipeline,
        }

    best_name = min(results, key=lambda k: results[k]['rmse'])
    payload = {
        'model_name': best_name,
        'model': results[best_name]['pipeline'],
        'metrics': {
            'mae': float(results[best_name]['mae']),
            'rmse': float(results[best_name]['rmse']),
            'r2': float(results[best_name]['r2']),
        },
        'comparisons': {name: {'mae': float(v['mae']), 'rmse': float(v['rmse']), 'r2': float(v['r2'])} for name, v in results.items()},
        'feature_columns': categorical + numeric,
    }
    joblib.dump(payload, model_path)
    return payload


def predict_yield(input_values: dict, model_path: Path | str | None = None):
    model_path = Path(model_path) if model_path is not None else MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError('Yield model not found. Run python utils/train_yield.py first.')

    payload = joblib.load(model_path)
    df = pd.DataFrame([input_values], columns=payload['feature_columns'])
    prediction = float(payload['model'].predict(df)[0])
    return {'yield_t_ha': prediction, 'metrics': payload['metrics']}


def validate_yield_inputs(raw_inputs: dict):
    errors = []
    required = ['crop', 'state', 'district', 'year', 'season', 'area_ha', 'rainfall_mm', 'temperature_c', 'soil_ph', 'soil_moisture_pct']
    for field in required:
        if field not in raw_inputs:
            errors.append(f'Missing required field: {field}')
            continue
        value = raw_inputs[field]
        if field in {'year', 'area_ha', 'rainfall_mm', 'temperature_c', 'soil_ph', 'soil_moisture_pct'}:
            try:
                value = float(value)
            except (TypeError, ValueError):
                errors.append(f'{field} must be a number.')
                continue
            if math.isnan(value) or math.isinf(value):
                errors.append(f'{field} must be a valid finite number.')
                continue
            if field == 'year' and value < 2000:
                errors.append('year must be valid.')
            if field == 'area_ha' and value <= 0:
                errors.append('area_ha must be greater than zero.')
            if field == 'soil_ph' and not 3 <= value <= 10:
                errors.append('soil_ph must be between 3 and 10.')
        elif field in {'crop', 'state', 'district', 'season'} and str(value).strip() == '':
            errors.append(f'{field} cannot be empty.')
    return errors


if __name__ == '__main__':
    ensure_yield_dataset(DATASET_PATH)
    payload = train_yield_model(DATASET_PATH, MODEL_PATH)
    print(json.dumps({'status': 'success', 'model_name': payload['model_name'], 'metrics': payload['metrics']}, indent=2))
