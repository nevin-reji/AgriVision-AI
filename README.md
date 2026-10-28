# AgriVision AI

AgriVision AI is a smart agricultural decision-support application built for a B.Tech AI/ML project. It includes three core modules:

- Crop recommendation using soil and weather parameters
- Crop yield prediction using historical agricultural features
- Plant disease detection using a MobileNetV2-based CNN pipeline

The application is implemented with Flask and trained using scikit-learn and TensorFlow.

## Project objectives

1. Recommend suitable crops from soil nutrient and climate conditions.
2. Predict crop yield using tabular agricultural data.
3. Classify plant leaf disease images with deep learning.
4. Provide a usable web dashboard for real predictions.
5. Maintain modular design and training scripts for future expansion.

## Technology stack

- Python 3.12
- Flask
- scikit-learn
- TensorFlow / Keras
- Pandas, NumPy
- Pillow
- Matplotlib
- Seaborn
- Jinja2 templates
- Pytest

## Project structure

- app.py — Flask application entry point
- requirements.txt — dependencies
- README.md — project documentation
- .env.example — sample environment values
- datasets/ — dataset storage
- models/ — saved model artifacts
- utils/ — training and preprocessing scripts
- templates/ — HTML pages
- static/ — CSS/JS assets
- tests/ — automated validation tests

## Installation

1. Open a terminal inside the project folder.
2. Create a virtual environment:
   python -m venv .venv
3. Activate it:
   .\.venv\Scripts\activate
4. Install dependencies:
   python -m pip install -r requirements.txt

## Running the app

Start the Flask dashboard:

flask --app app run --debug

Then open:

http://127.0.0.1:5000/

## Training commands

Crop recommendation:
python utils/train_crop.py

Yield prediction:
python utils/train_yield.py

Disease detection:
python utils/train_disease.py

## Dataset notes

The project is designed to work with real public datasets when they are available. If the dataset is missing, the project creates a local demo dataset so the modules can still be trained and tested.

Important:
- Crop recommendation uses the dataset path `datasets/crop_recommendation.csv`
- Yield dataset uses `datasets/crop_yield.csv`
- Disease dataset uses `datasets/plant_disease/`

The demo datasets are meant for local execution and learning; real-world agricultural use should rely on actual field and public benchmark data.

## Model summary

Crop recommendation model:
- Random Forest Classifier
- Saved as `models/crop_model.joblib`

Yield prediction model:
- Best-performing regressor chosen from Linear Regression and Random Forest Regressor
- Saved as `models/yield_model.joblib`

Disease detection model:
- MobileNetV2 transfer learning
- Saved as `models/disease_model.keras`
- Class mapping stored in `models/disease_labels.json`

## Testing

Run the project tests:

python -m pytest -q

The suite covers:
- crop recommendation training
- yield prediction training
- disease model training
- input validation
- route smoke checks

## Usage

1. Open the homepage.
2. Choose the Crop Recommendation page to input N, P, K, temperature, humidity, pH, and rainfall.
3. Choose the Yield page to estimate yield using agricultural features.
4. Choose the Disease page to upload a leaf image and get a prediction.

## Limitations

- Demo datasets are used when real data is absent.
- The disease model is designed for educational use and not as a substitute for certified agronomic diagnosis.
- Real-world agricultural models should be validated with field experts and richer datasets.

## Future work

- Add SQLite-based prediction history
- Integrate more real medical/agronomic datasets
- Add model comparison charts and evaluation dashboards
- Expand disease categories and farmer advisory features
- Deploy the project on a cloud host

## College project report outline

1. Abstract
2. Introduction
3. Literature review
4. Problem statement
5. Existing system
6. Proposed system
7. Methodology
8. Architecture
9. Implementation
10. Results and discussion
11. Testing
12. Limitations
13. Future scope
14. Conclusion
15. References
