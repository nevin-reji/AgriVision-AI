import numpy as np
from PIL import Image

from utils.train_disease import create_demo_disease_dataset, predict_disease, train_disease_model


def test_demo_disease_dataset_creation(tmp_path):
    dataset_dir = tmp_path / 'plant_disease'
    create_demo_disease_dataset(dataset_dir, samples_per_class=10)
    assert dataset_dir.exists()
    assert any(p.is_dir() for p in dataset_dir.iterdir())


def test_train_disease_model_and_prediction(tmp_path):
    dataset_dir = tmp_path / 'plant_disease'
    model_path = tmp_path / 'disease_model.keras'
    labels_path = tmp_path / 'disease_labels.json'
    create_demo_disease_dataset(dataset_dir, samples_per_class=12)
    payload = train_disease_model(dataset_dir, model_path, labels_path)
    assert 'classes' in payload
    assert 'metrics' in payload

    sample_image = Image.new('RGB', (224, 224), color=(80, 200, 80))
    sample_path = tmp_path / 'sample.png'
    sample_image.save(sample_path)
    result = predict_disease(sample_path, model_path, labels_path)
    assert 'label' in result
    assert 'confidence' in result
    assert isinstance(result['confidence'], float)
