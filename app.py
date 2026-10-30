
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("dog_cnn.keras")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None

    if request.method == "POST":

        file = request.files.get("image")

        if file and file.filename:

            # Read uploaded image
            image = Image.open(
                BytesIO(file.read())
            )

            # Convert image to RGB
            image = image.convert("RGB")

            # Resize image
            image = image.resize((128, 128))

            # Convert image to NumPy array
            image_array = np.array(
                image,
                dtype=np.float32
            )

            # Add batch dimension
            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # IMPORTANT:
            # Do not divide by 255.
            # MobileNetV2 preprocessing is already
            # included inside the trained model.

            score = float(
                model.predict(
                    image_array,
                    verbose=0
                )[0][0]
            )

            # Class prediction
            if score >= 0.5:

                prediction = "Dog"
                confidence = score * 100

            else:

                prediction = "Cat"
                confidence = (1 - score) * 100

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)
