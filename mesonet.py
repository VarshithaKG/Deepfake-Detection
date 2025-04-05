import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load MesoNet model
MODEL_PATH = "Meso4_DF.h5"

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print("Error loading model:", e)

# Function to predict deepfake
def predict_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(256, 256))
        img_array = image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)

        # Model prediction
        prediction = model.predict(img_array)[0][0]  # Assuming binary classification

        if prediction > 0.5:
            return "FAKE", round(prediction * 100, 2)
        else:
            return "REAL", round((1 - prediction) * 100, 2)

    except Exception as e:
        print("Prediction error:", e)
        return "ERROR", 0
