import pandas as pd

from utils.train_crop import create_demo_crop_dataset, predict_crop, train_crop_model


def test_demo_crop_dataset_creation(tmp_path):
    output = tmp_path / 'demo_crop_recommendation.csv'
    create_demo_crop_dataset(output)
    df = pd.read_csv(output)
    assert 'label' in df.columns
    assert len(df) > 0
    assert 'N' in df.columns


def test_train_crop_model_and_prediction(tmp_path):
    dataset_path = tmp_path / 'crop_recommendation.csv'
    model_path = tmp_path / 'crop_model.joblib'
    create_demo_crop_dataset(dataset_path)
    payload = train_crop_model(dataset_path, model_path)
    assert 'pipeline' in payload
    assert 'metrics' in payload

    result = predict_crop({'N': 45, 'P': 50, 'K': 30, 'temperature': 24, 'humidity': 60, 'ph': 6.0, 'rainfall': 80}, model_path=model_path)
    assert 'prediction' in result
    assert 'confidence' in result
    assert isinstance(result['prediction'], str)
