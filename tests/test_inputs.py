from utils.train_crop import validate_crop_inputs


def test_invalid_numeric_inputs():
    errors = validate_crop_inputs({'N': -2, 'P': 10, 'K': 12, 'temperature': 25, 'humidity': 60, 'ph': 6.5, 'rainfall': 80})
    assert any('N' in err for err in errors)
