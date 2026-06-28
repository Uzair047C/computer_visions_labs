import json
import random
import shutil
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

BASE_DIR = Path(__file__).parent
SOURCE_DIR = BASE_DIR / "hotdog_nothotdog" / "transfer-learning-with-cnn-15-2025-06-24-t-12-35-01-829-z"
DATASET_DIR = BASE_DIR / "dataset_simple"


def make_train_val_folders(source_dir, output_dir, train_ratio=0.9):
    """Create train/val class folders from the annotation JSON."""
    annotations_file = source_dir / "_annotations.json"

    with annotations_file.open("r") as f:
        annotations = json.load(f)["annotations"]

    label_to_files = {}
    for filename, entry in annotations.items():
        label = entry[0]["label"]
        label_to_files.setdefault(label, []).append(filename)

    for label, files in label_to_files.items():
        random.shuffle(files)
        split_at = int(len(files) * train_ratio)
        splits = {
            "train": files[:split_at],
            "val": files[split_at:],
        }

        for split_name, split_files in splits.items():
            class_dir = output_dir / split_name / label
            class_dir.mkdir(parents=True, exist_ok=True)

            for filename in split_files:
                shutil.copy2(source_dir / filename, class_dir / filename)


def load_dataset(folder, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        folder,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )


def prepare_dataset(dataset, augment=False):
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(5 / 360),
    ])

    def prepare(images, labels):
        if augment:
            images = augmentation(images, training=True)
        images = preprocess_input(images)
        return images, labels

    return dataset.map(prepare).prefetch(tf.data.AUTOTUNE)


def build_resnet50_model(num_classes):
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes)(x)

    return Model(inputs, outputs)


def main():
    random.seed(0)
    tf.random.set_seed(0)

    make_train_val_folders(SOURCE_DIR, DATASET_DIR)

    train_raw = load_dataset(DATASET_DIR / "train", shuffle=True)
    val_raw = load_dataset(DATASET_DIR / "val", shuffle=False)

    class_names = train_raw.class_names
    train_data = prepare_dataset(train_raw, augment=True)
    val_data = prepare_dataset(val_raw, augment=False)

    model = build_resnet50_model(num_classes=len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(train_data, validation_data=val_data, epochs=EPOCHS)
    model.save_weights(BASE_DIR / "simple_resnet50.weights.h5")

    print("Classes:", class_names)
    print("Saved weights to simple_resnet50.weights.h5")


if __name__ == "__main__":
    main()
