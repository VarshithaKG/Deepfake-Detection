import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense

# Define Meso4 model
def Meso4():
    x = Input(shape=(256, 256, 3))
    y = Conv2D(8, (3, 3), padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = MaxPooling2D(pool_size=(2, 2), padding='same')(y)

    y = Conv2D(8, (5, 5), padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = MaxPooling2D(pool_size=(2, 2), padding='same')(y)

    y = Conv2D(16, (5, 5), padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = MaxPooling2D(pool_size=(2, 2), padding='same')(y)

    y = Conv2D(16, (5, 5), padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = MaxPooling2D(pool_size=(4, 4), padding='same')(y)

    y = Flatten()(y)
    y = Dense(16)(y)
    y = Activation('relu')(y)
    y = Dense(1, activation='sigmoid')(y)

    return Model(inputs=x, outputs=y)

# Load model
model = Meso4()
model.load_weights("Meso4_DF.h5")

# Flask app setup
app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    filepath = os.path.join('static/uploads', file.filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file.save(filepath)

    # Image preprocessing
    img = cv2.imread(filepath)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction and confidence
    prediction_score = model.predict(img)[0][0]
    confidence = prediction_score * 100 if prediction_score > 0.5 else (1 - prediction_score) * 100
    result = "Real" if prediction_score > 0.5 else "Deepfake"
    color = "green" if result == "Real" else "red"

    return render_template('result.html',
                           prediction=result,
                           confidence=confidence,
                           filename=file.filename,
                           color=color)

if __name__ == '__main__':
    app.run(debug=True)
