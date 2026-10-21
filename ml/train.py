#!/usr/bin/env python3
"""Train a proper MNIST CNN and export a TF.js model the live demo can load.

Usage:
    python3 -m venv .venv && . .venv/bin/activate
    pip install -r ml/requirements.txt
    python ml/train.py            # trains, evaluates, exports to ./model/

Output:
    model/model.json + model/group1-shard*.bin   (TF.js Layers format, loaded by app.js)
    ml/metrics.json                                (true held-out test metrics)
"""
import json, os
import numpy as np

def main():
    import tensorflow as tf
    from tensorflow import keras

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32")[..., None] / 255.0
    x_test = x_test.astype("float32")[..., None] / 255.0
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model = keras.Sequential([
        keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(64, 3, activation="relu"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    cb = [keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)]
    hist = model.fit(x_train, y_train, batch_size=128, epochs=12,
                     validation_split=0.1, callbacks=cb, verbose=2)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"TEST accuracy={acc:.4f} loss={loss:.4f}")

    os.makedirs("model", exist_ok=True)
    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(model, "model")

    metrics = {
        "test_accuracy": round(float(acc), 4),
        "test_loss": round(float(loss), 4),
        "test_samples": int(len(x_test)),
        "train_samples": int(len(x_train)),
        "epochs_trained": len(hist.history["loss"]),
        "framework": "TensorFlow (Keras) + tensorflowjs converter",
        "dataset": "MNIST (keras.datasets.mnist, 60k train / 10k test)",
    }
    os.makedirs("ml", exist_ok=True)
    with open("ml/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Exported model/ and ml/metrics.json")

if __name__ == "__main__":
    main()
