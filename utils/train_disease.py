import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from utils import BASE_DIR

DATASET_ROOT = BASE_DIR / 'datasets' / 'plant_disease_real'
MODEL_DIR = BASE_DIR / 'models'
MODEL_PATH = MODEL_DIR / 'disease_model.keras'
LABELS_PATH = MODEL_DIR / 'disease_labels.json'

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 6
SEED = 42

MODEL_DIR.mkdir(parents=True, exist_ok=True)
tf.keras.utils.set_random_seed(SEED)


def create_demo_disease_dataset(output_dir: Path | str | None = None, samples_per_class: int = 12) -> Path:
    output_dir = Path(output_dir) if output_dir is not None else BASE_DIR / 'datasets' / 'plant_disease'
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = ['healthy', 'early_blight', 'late_blight', 'leaf_spot', 'rust']
    palettes = {
        'healthy': (70, 200, 80),
        'early_blight': (180, 110, 40),
        'late_blight': (160, 60, 60),
        'leaf_spot': (120, 170, 50),
        'rust': (210, 130, 20),
    }

    for class_name in class_names:
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(samples_per_class):
            image = Image.new('RGB', IMAGE_SIZE, color=palettes[class_name])
            draw = ImageDraw.Draw(image)
            for i in range(8):
                x = 20 + (i * 24) + (index % 3) * 8
                y = 40 + ((i * 18) % 150)
                radius = 12 + (i % 5) * 4
                fill = (
                    max(0, palettes[class_name][0] - 25 + i * 8),
                    max(0, palettes[class_name][1] - 20 + i * 6),
                    max(0, palettes[class_name][2] - 15 + i * 5),
                )
                draw.ellipse((x, y, x + radius * 2, y + radius * 2), outline=fill, width=3)

            if class_name == 'healthy':
                draw.rectangle((40, 40, 180, 180), outline=(25, 120, 40), width=5)
            elif class_name == 'early_blight':
                draw.line((30, 180, 200, 60), fill=(90, 60, 40), width=4)
            elif class_name == 'late_blight':
                draw.polygon([(40, 180), (120, 30), (200, 180)], outline=(120, 30, 30), width=4)
            elif class_name == 'leaf_spot':
                for spot_x in range(50, 200, 30):
                    draw.ellipse((spot_x, 50, spot_x + 18, 85), fill=(80, 120, 40))
            else:
                for rust_x in range(30, 210, 26):
                    draw.ellipse((rust_x, 50, rust_x + 24, 80), fill=(180, 90, 10))

            image.save(class_dir / f'{class_name}_{index}.png')

    return output_dir


def ensure_disease_dataset(dataset_root: Path | str | None = None) -> Path:
    dataset_root = Path(dataset_root) if dataset_root is not None else DATASET_ROOT
    dataset_root = dataset_root.expanduser().resolve()
    if not dataset_root.exists():
        fallback = BASE_DIR / 'datasets' / 'plant_disease'
        if fallback.exists():
            return fallback
        return create_demo_disease_dataset(dataset_root)
    return dataset_root


def _load_image_dataset(dataset_root: Path | str | None = None):
    dataset_root = ensure_disease_dataset(dataset_root)
    class_names = sorted(folder.name for folder in dataset_root.iterdir() if folder.is_dir())
    if not class_names:
        raise ValueError(f'No class folders found under {dataset_root}')

    image_paths = []
    labels = []
    for class_index, class_name in enumerate(class_names):
        class_dir = dataset_root / class_name
        files = [
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
        ]
        if not files:
            continue
        image_paths.extend(str(path) for path in files)
        labels.extend([class_index] * len(files))

    if not image_paths:
        raise ValueError(f'No image files found under {dataset_root}')

    return np.array(image_paths), np.array(labels), class_names


def _load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label


def _create_tf_dataset(paths, targets, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((paths, targets))
    if training:
        dataset = dataset.shuffle(len(paths), seed=SEED)
    dataset = dataset.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def train_disease_model(dataset_root: Path | str | None = None, model_path: Path | str | None = None, labels_path: Path | str | None = None):
    dataset_root = ensure_disease_dataset(dataset_root)
    model_path = Path(model_path) if model_path is not None else MODEL_PATH
    labels_path = Path(labels_path) if labels_path is not None else LABELS_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths, labels, class_names = _load_image_dataset(dataset_root)
    if len(class_names) < 2:
        raise ValueError(f'At least two classes are needed for disease classification: {dataset_root}')

    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths,
        labels,
        test_size=0.30,
        random_state=SEED,
        stratify=labels,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=SEED,
        stratify=y_temp,
    )

    train_ds = _create_tf_dataset(X_train, y_train, training=True)
    val_ds = _create_tf_dataset(X_val, y_val)
    test_ds = _create_tf_dataset(X_test, y_test)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=0,
    )

    model.save(model_path)
    with labels_path.open('w', encoding='utf-8') as file:
        json.dump({str(index): name for index, name in enumerate(class_names)}, file, indent=4)

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    y_prob = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    metrics = {
        'loss': float(test_loss),
        'accuracy': float(test_accuracy),
        'val_accuracy': float(np.max(history.history.get('val_accuracy', [0.0]))),
        'classes': class_names,
    }

    return {'model': model, 'classes': class_names, 'metrics': metrics}


def predict_disease(image_path: Path | str, model_path: Path | str | None = None, labels_path: Path | str | None = None):
    model_path = Path(model_path) if model_path is not None else MODEL_PATH
    labels_path = Path(labels_path) if labels_path is not None else LABELS_PATH

    if not model_path.exists():
        train_disease_model(model_path=model_path, labels_path=labels_path)

    with labels_path.open('r', encoding='utf-8') as file:
        labels = json.load(file)

    model = tf.keras.models.load_model(model_path)
    image = Image.open(image_path).convert('RGB').resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array[np.newaxis, ...])
    probs = model.predict(image_array, verbose=0)[0]
    class_index = int(np.argmax(probs))
    label = labels.get(str(class_index), str(class_index))
    confidence = float(probs[class_index])
    return {'label': label, 'confidence': confidence, 'class_index': class_index, 'probabilities': {key: float(value) for key, value in zip(labels.values(), probs)}}


DATASET_ROOT = ensure_disease_dataset(DATASET_ROOT)
