import json
import math
import random
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import BASE_DIR, CROP_PROFILES, FEATURE_COLUMNS

DATASET_PATH = BASE_DIR / 'datasets' / 'crop_recommendation.csv'
MODEL_PATH = BASE_DIR / 'models' / 'crop_model.joblib'


def create_demo_crop_dataset(output_path: Path | None = None) -> Path:
    output_path = output_path or DATASET_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    random.seed(42)

    for crop_name, profile in CROP_PROFILES.items():
        for _ in range(100):
            row = {
                'N': int(profile['N'] + random.uniform(-12, 12)),
                'P': int(profile['P'] + random.uniform(-10, 10)),
                'K': int(profile['K'] + random.uniform(-10, 10)),
                'temperature': round(profile['temperature'] + random.uniform(-5, 5), 2),
                'humidity': round(profile['humidity'] + random.uniform(-10, 10), 2),
                'ph': round(profile['ph'] + random.uniform(-0.8, 0.8), 2),
                'rainfall': round(profile['rainfall'] + random.uniform(-25, 25), 2),
                'label': crop_name,
            }
            records.append(row)

    df = pd.DataFrame(records)
    df = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']]
    df.to_csv(output_path, index=False)
    return output_path


def ensure_crop_dataset(dataset_path: Path | None = None) -> Path:
    dataset_path = dataset_path or DATASET_PATH
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if not dataset_path.exists():
        print(f'Warning: dataset not found at {dataset_path}. Creating a demo dataset for local testing.')
        return create_demo_crop_dataset(dataset_path)
    return dataset_path


def _build_pipeline():
    preprocessor = ColumnTransformer(
        [('numeric', StandardScaler(), FEATURE_COLUMNS)],
        remainder='drop',
    )
    classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced',
    )
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier),
    ])
    return pipeline


def train_crop_model(dataset_path: Path | None = None, model_path: Path | None = None):
    dataset_path = ensure_crop_dataset(dataset_path or DATASET_PATH)
    model_path = model_path or MODEL_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    if 'label' not in df.columns:
        raise ValueError('The crop dataset must contain a label column named "label".')

    X = df[FEATURE_COLUMNS]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average='macro', zero_division=0
    )

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'dataset_source': 'demo',
    }

    payload = {
        'pipeline': pipeline,
        'feature_names': FEATURE_COLUMNS,
        'classes': sorted(y.unique().tolist()),
        'metrics': metrics,
    }
    joblib.dump(payload, model_path)
    return payload


def predict_crop(input_values: dict, model_path: Path | str | None = None):
    model_path = Path(model_path) if model_path is not None else MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError('Crop model not found. Run python utils/train_crop.py first.')

    payload = joblib.load(model_path)
    row = {field: float(input_values[field]) for field in FEATURE_COLUMNS}
    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    prediction = payload['pipeline'].predict(df)[0]
    probabilities = payload['pipeline'].predict_proba(df)[0]
    confidence = float(probabilities.max())
    return {'prediction': prediction, 'confidence': confidence, 'metrics': payload['metrics']}


def validate_crop_inputs(raw_inputs: dict):
    errors = []
    for field in FEATURE_COLUMNS:
        if field not in raw_inputs:
            errors.append(f'Missing required field: {field}')
            continue

        try:
            value = float(raw_inputs[field])
        except (TypeError, ValueError):
            errors.append(f'{field} must be numeric.')
            continue

        if math.isnan(value) or math.isinf(value):
            errors.append(f'{field} must be a valid finite number.')
            continue

        if field in {'N', 'P', 'K'} and value < 0:
            errors.append(f'{field} must be greater than or equal to zero.')

        if field == 'ph' and not 0 <= value <= 14:
            errors.append('ph must be between 0 and 14.')

        if field in {'temperature', 'humidity', 'rainfall'} and value < 0:
            errors.append(f'{field} must be greater than or equal to zero.')

    return errors


if __name__ == '__main__':
    ensure_crop_dataset(DATASET_PATH)
    payload = train_crop_model(DATASET_PATH, MODEL_PATH)
    print(json.dumps({'status': 'success', 'metrics': payload['metrics']}, indent=2))
