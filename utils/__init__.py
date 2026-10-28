from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
CROP_PROFILES = {
    'rice': {'N': 80, 'P': 60, 'K': 40, 'temperature': 25, 'humidity': 82, 'ph': 6.0, 'rainfall': 200},
    'maize': {'N': 78, 'P': 48, 'K': 40, 'temperature': 22, 'humidity': 65, 'ph': 5.8, 'rainfall': 120},
    'chickpea': {'N': 40, 'P': 70, 'K': 20, 'temperature': 18, 'humidity': 55, 'ph': 5.8, 'rainfall': 80},
    'kidneybeans': {'N': 20, 'P': 90, 'K': 30, 'temperature': 20, 'humidity': 60, 'ph': 6.8, 'rainfall': 100},
    'pigeonpeas': {'N': 20, 'P': 60, 'K': 20, 'temperature': 22, 'humidity': 65, 'ph': 6.0, 'rainfall': 120},
    'mothbeans': {'N': 20, 'P': 50, 'K': 20, 'temperature': 26, 'humidity': 60, 'ph': 6.2, 'rainfall': 35},
    'mungbean': {'N': 20, 'P': 50, 'K': 45, 'temperature': 28, 'humidity': 70, 'ph': 5.5, 'rainfall': 65},
    'blackgram': {'N': 40, 'P': 60, 'K': 20, 'temperature': 30, 'humidity': 70, 'ph': 6.0, 'rainfall': 110},
    'lentil': {'N': 20, 'P': 70, 'K': 15, 'temperature': 16, 'humidity': 55, 'ph': 6.5, 'rainfall': 70},
    'pomegranate': {'N': 20, 'P': 60, 'K': 30, 'temperature': 28, 'humidity': 70, 'ph': 5.5, 'rainfall': 90},
    'banana': {'N': 100, 'P': 50, 'K': 90, 'temperature': 27, 'humidity': 80, 'ph': 6.0, 'rainfall': 120},
    'mango': {'N': 20, 'P': 40, 'K': 30, 'temperature': 30, 'humidity': 75, 'ph': 6.5, 'rainfall': 120},
    'grapes': {'N': 30, 'P': 40, 'K': 50, 'temperature': 24, 'humidity': 80, 'ph': 6.0, 'rainfall': 100},
    'watermelon': {'N': 20, 'P': 25, 'K': 15, 'temperature': 30, 'humidity': 65, 'ph': 6.5, 'rainfall': 120},
    'muskmelon': {'N': 18, 'P': 35, 'K': 20, 'temperature': 29, 'humidity': 65, 'ph': 6.2, 'rainfall': 80},
    'apple': {'N': 20, 'P': 30, 'K': 30, 'temperature': 20, 'humidity': 78, 'ph': 6.1, 'rainfall': 100},
    'orange': {'N': 15, 'P': 30, 'K': 20, 'temperature': 25, 'humidity': 75, 'ph': 6.4, 'rainfall': 80},
    'papaya': {'N': 30, 'P': 60, 'K': 20, 'temperature': 28, 'humidity': 72, 'ph': 6.0, 'rainfall': 140},
    'coconut': {'N': 30, 'P': 40, 'K': 40, 'temperature': 28, 'humidity': 80, 'ph': 6.0, 'rainfall': 200},
    'cotton': {'N': 80, 'P': 50, 'K': 30, 'temperature': 25, 'humidity': 70, 'ph': 5.8, 'rainfall': 80},
    'jute': {'N': 80, 'P': 60, 'K': 40, 'temperature': 26, 'humidity': 80, 'ph': 6.1, 'rainfall': 140},
    'coffee': {'N': 30, 'P': 35, 'K': 35, 'temperature': 22, 'humidity': 75, 'ph': 6.2, 'rainfall': 120},
}
FEATURE_COLUMNS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
