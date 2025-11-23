import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import requests
from io import BytesIO

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model.keras")
LABELS_PATH = os.path.join(BASE_DIR, "..", "models", "labels.json")

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    model = None

# Load labels
try:
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
    print("Labels loaded successfully.")
except Exception as e:
    print("Error loading labels:", e)
    labels = None

# --- IMPORTANT: Use ConvNeXt preprocess ---
from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess

def preprocess_image(image: Image.Image) -> np.ndarray:
    target_size = (224, 224)

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target_size)
    img_array = tf.keras.utils.img_to_array(image)

    img_array = tf.expand_dims(img_array, axis=0)
    img_array = convnext_preprocess(img_array)

    return img_array

def predict_dog_breed_from_url(image_url: str) -> int:
    if model is None or labels is None:
        raise Exception("Model or labels not loaded.")

    # Download image from URL
    response = requests.get(image_url)
    response.raise_for_status()
    
    image = Image.open(BytesIO(response.content))
    processed_image = preprocess_image(image)
    preds = model.predict(processed_image)[0]

    # Return the label number (index)
    label_number = int(np.argmax(preds))
    return label_number

# Keep the old function for backward compatibility
def predict_dog_breed(image: Image.Image) -> str:
    if model is None or labels is None:
        return "Model or labels not loaded."

    processed_image = preprocess_image(image)
    preds = model.predict(processed_image)[0]

    index = int(np.argmax(preds))
    raw_name = labels.get(str(index), "Unknown")

    # Clean "n02085620-Chihuahua"
    if "-" in raw_name:
        raw_name = raw_name.split("-", 1)[1]

    clean = raw_name.replace("_", " ").title()
    return clean
