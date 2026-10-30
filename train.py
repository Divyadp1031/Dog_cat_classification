
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------- CONFIGURATION ----------------

MODE = "transfer"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 20

CLASS_NAMES = ["cats", "dogs"]
SEED = 123

ROOT_DIR = Path(__file__).resolve().parent

TRAIN_DIR = ROOT_DIR / "dataset" / "training_set"
TEST_DIR = ROOT_DIR / "dataset" / "test_set"

MODEL_PATH = ROOT_DIR / "dog_cnn.keras"

AUTOTUNE = tf.data.AUTOTUNE

tf.keras.utils.set_random_seed(SEED)


# ---------------- CHECK DATASET ----------------

if not TRAIN_DIR.exists():
    raise FileNotFoundError(f"Training folder not found: {TRAIN_DIR}")

if not TEST_DIR.exists():
    raise FileNotFoundError(f"Testing folder not found: {TEST_DIR}")

print("Available GPUs:", tf.config.list_physical_devices("GPU"))


# ---------------- LOAD TRAINING DATA ----------------

def load_data(subset):

    dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        validation_split=0.2,
        subset=subset,
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=(subset == "training")
    )

    return dataset


train_data = load_data("training")
validation_data = load_data("validation")


# ---------------- LOAD TEST DATA ----------------

test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# Optimize data loading without caching the entire dataset in RAM.

train_data = train_data.prefetch(1)
validation_data = validation_data.prefetch(1)
test_data = test_data.prefetch(1)


# ---------------- DATA AUGMENTATION ----------------

augmentation = models.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
    ],
    name="augmentation"
)


# ---------------- BUILD MODEL ----------------

def build_model():

    inputs = layers.Input(shape=(128, 128, 3))

    # Augmentation during training only
    x = augmentation(inputs)

    # Pretrained MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze pretrained layers
    base_model.trainable = False

    # MobileNetV2 preprocessing
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # Feature extraction
    x = base_model(x, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.3)(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    model = models.Model(inputs, outputs)

    return model


model = build_model()

model.summary()


# ---------------- COMPILE MODEL ----------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ---------------- CALLBACKS ----------------

callbacks = [

    # Stop when validation performance stops improving
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    # Reduce learning rate when progress slows
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=0.000001,
        verbose=1
    ),

    # Save the best model
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]


# ---------------- TRAIN MODEL ----------------

print("\nStarting model training...\n")

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)


# ---------------- LOAD BEST MODEL ----------------

best_model = tf.keras.models.load_model(MODEL_PATH)


# ---------------- EVALUATE MODEL ----------------

print("\nEvaluating model on test dataset...\n")

test_loss, test_accuracy = best_model.evaluate(
    test_data,
    verbose=1
)

print("\n========== MODEL RESULTS ==========")

print(f"Epochs completed: {len(history.epoch)}")

print(f"Test Loss: {test_loss:.4f}")

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print(f"Model saved at: {MODEL_PATH}")

print("Class mapping:", CLASS_NAMES)

print("===================================")
