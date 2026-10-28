import pandas as pd

from utils.train_yield import create_demo_yield_dataset, predict_yield, train_yield_model


def test_demo_yield_dataset_creation(tmp_path):
    output = tmp_path / 'crop_yield.csv'
    create_demo_yield_dataset(output)
    df = pd.read_csv(output)
    assert 'yield_t_ha' in df.columns
    assert len(df) > 0
    assert 'crop' in df.columns


def test_train_yield_model_and_prediction(tmp_path):
    dataset_path = tmp_path / 'crop_yield.csv'
    model_path = tmp_path / 'yield_model.joblib'
    create_demo_yield_dataset(dataset_path)
    payload = train_yield_model(dataset_path, model_path)
    assert 'model' in payload
    assert 'metrics' in payload

    result = predict_yield({
        'crop': 'rice',
        'state': 'Punjab',
        'district': 'Punjab-1',
        'year': 2023,
        'season': 'Kharif',
        'area_ha': 1200,
        'rainfall_mm': 850,
        'temperature_c': 29,
        'soil_ph': 6.5,
        'soil_moisture_pct': 45,
    }, model_path=model_path)
    assert 'yield_t_ha' in result
    assert isinstance(result['yield_t_ha'], float)
