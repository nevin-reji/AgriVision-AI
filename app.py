from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from utils.train_crop import DATASET_PATH as CROP_DATASET_PATH, MODEL_PATH as CROP_MODEL_PATH, predict_crop, train_crop_model, validate_crop_inputs
from utils.train_disease import DATASET_ROOT as DISEASE_DATASET_ROOT, MODEL_PATH as DISEASE_MODEL_PATH, LABELS_PATH as DISEASE_LABELS_PATH, predict_disease, train_disease_model
from utils.train_yield import DATASET_PATH as YIELD_DATASET_PATH, MODEL_PATH as YIELD_MODEL_PATH, predict_yield, train_yield_model, validate_yield_inputs

app = Flask(__name__)
FEATURE_FIELDS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
YIELD_FIELDS = ['crop', 'state', 'district', 'year', 'season', 'area_ha', 'rainfall_mm', 'temperature_c', 'soil_ph', 'soil_moisture_pct']
ALLOWED_IMAGE_TYPES = {'image/png', 'image/jpeg', 'image/jpg'}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/crop', methods=['GET', 'POST'])
def crop_page():
    model_ready = CROP_MODEL_PATH.exists()
    payload = {field: '' for field in FEATURE_FIELDS}

    if request.method == 'POST':
        payload = {field: request.form.get(field, '') for field in FEATURE_FIELDS}
        errors = validate_crop_inputs(payload)
        if errors:
            return render_template('crop.html', model_ready=model_ready, fields=payload, error=' '.join(errors))

        if not CROP_MODEL_PATH.exists():
            train_crop_model(CROP_DATASET_PATH, CROP_MODEL_PATH)

        try:
            result = predict_crop(payload)
            return render_template(
                'crop.html',
                model_ready=True,
                fields=payload,
                prediction=result['prediction'],
                confidence=result['confidence'],
                metrics=result['metrics'],
            )
        except Exception as exc:
            return render_template('crop.html', model_ready=model_ready, fields=payload, error=str(exc))

    return render_template('crop.html', model_ready=model_ready, fields=payload)


@app.route('/yield', methods=['GET', 'POST'])
def yield_page():
    model_ready = YIELD_MODEL_PATH.exists()
    payload = {field: '' for field in YIELD_FIELDS}

    if request.method == 'POST':
        payload = {field: request.form.get(field, '') for field in YIELD_FIELDS}
        errors = validate_yield_inputs(payload)
        if errors:
            return render_template('yield.html', model_ready=model_ready, fields=payload, error=' '.join(errors))

        if not YIELD_MODEL_PATH.exists():
            train_yield_model(YIELD_DATASET_PATH, YIELD_MODEL_PATH)

        try:
            result = predict_yield(payload)
            return render_template(
                'yield.html',
                model_ready=True,
                fields=payload,
                prediction=result['yield_t_ha'],
                metrics=result['metrics'],
            )
        except Exception as exc:
            return render_template('yield.html', model_ready=model_ready, fields=payload, error=str(exc))

    return render_template('yield.html', model_ready=model_ready, fields=payload)


@app.route('/history')
def history_page():
    return render_template('history.html')


@app.route('/disease', methods=['GET', 'POST'])
def disease_page():
    model_ready = DISEASE_MODEL_PATH.exists() and DISEASE_LABELS_PATH.exists()
    result = None

    if request.method == 'POST':
        file = request.files.get('image')
        if not file or not file.filename:
            return render_template('disease.html', model_ready=model_ready, error='Please upload an image file.')

        if file.mimetype not in ALLOWED_IMAGE_TYPES:
            return render_template('disease.html', model_ready=model_ready, error='Unsupported file type. Please upload PNG or JPEG.')

        if not DISEASE_MODEL_PATH.exists():
            train_disease_model(DISEASE_DATASET_ROOT, DISEASE_MODEL_PATH, DISEASE_LABELS_PATH)
        try:
            filename = secure_filename(file.filename)
            temp_path = f'tmp_{filename}'
            file.save(temp_path)
            result = predict_disease(temp_path, DISEASE_MODEL_PATH, DISEASE_LABELS_PATH)
            return render_template('disease.html', model_ready=True, result=result, image_name=filename)
        except Exception as exc:
            return render_template('disease.html', model_ready=model_ready, error=str(exc))

    return render_template('disease.html', model_ready=model_ready, result=result)


@app.route('/api/crop/predict', methods=['POST'])
def predict_api():
    payload = request.get_json(silent=True) or {}
    errors = validate_crop_inputs(payload)
    if errors:
        return jsonify({'success': False, 'errors': errors}), 400

    if not CROP_MODEL_PATH.exists():
        train_crop_model(CROP_DATASET_PATH, CROP_MODEL_PATH)

    try:
        result = predict_crop(payload)
        return jsonify({'success': True, 'prediction': result['prediction'], 'confidence': result['confidence']})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/yield/predict', methods=['POST'])
def predict_yield_api():
    payload = request.get_json(silent=True) or {}
    errors = validate_yield_inputs(payload)
    if errors:
        return jsonify({'success': False, 'errors': errors}), 400

    if not YIELD_MODEL_PATH.exists():
        train_yield_model(YIELD_DATASET_PATH, YIELD_MODEL_PATH)

    try:
        result = predict_yield(payload)
        return jsonify({'success': True, 'yield_t_ha': result['yield_t_ha'], 'metrics': result['metrics']})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.errorhandler(404)
def page_not_found(_):
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(_):
    return render_template('500.html'), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
