import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import database FIRST to avoid conflict with TensorFlow
# from ..database import SessionLocal
# from ..models.sql_models import ImageLog

import tensorflow as tf
import numpy as np
from PIL import Image
import json
import requests
from io import BytesIO
import hashlib

# --- IMPORTANT: Use ConvNeXt preprocess ---
from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess

from .dog_detection import detect_dog

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


def preprocess_image(image: Image.Image) -> np.ndarray:
    target_size = (224, 224)

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target_size)
    img_array = tf.keras.utils.img_to_array(image)

    img_array = tf.expand_dims(img_array, axis=0)
    img_array = convnext_preprocess(img_array)

    return img_array

def compute_image_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()

def predict_dog_breed_from_url(image_url: str) -> int:
    if model is None or labels is None:
        raise Exception("Model or labels not loaded.")

    # Download image from URL
    response = requests.get(image_url)
    response.raise_for_status()
    image_bytes = response.content
    
    # Compute Hash
    image_hash = compute_image_hash(image_bytes)

    # Check Database
    from ..database import get_supabase_client
    supabase = get_supabase_client()
    
    if supabase:
        try:
            response = supabase.table("image_logs").select("*").eq("image_hash", image_hash).execute()
            if response.data:
                cached_result = response.data[0]
                print(f"Cache hit for {image_hash}")
                if not cached_result["is_dog"]:
                     raise Exception("No dog detected in the image (Cached).")
                return cached_result["breed_label"]
        except Exception as e:
            print(f"Error querying Supabase: {e}")

    # Process Image
    image = Image.open(BytesIO(image_bytes))
    
    # 1. Detect Dog using YOLO
    if not detect_dog(image):
        # Save negative result to DB
        if supabase:
            try:
                new_log = {
                    "image_hash": image_hash,
                    "image_url": image_url,
                    "is_dog": False,
                    "breed_label": None
                }
                supabase.table("image_logs").insert(new_log).execute()
            except Exception as e:
                print(f"Error saving to DB: {e}")
            
        raise Exception("No dog detected in the image.")

    # 2. Predict Breed
    processed_image = preprocess_image(image)
    preds = model.predict(processed_image)[0]
    label_number = int(np.argmax(preds))

    # Save positive result to DB
    if supabase:
        try:
            new_log = {
                "image_hash": image_hash,
                "image_url": image_url,
                "is_dog": True,
                "breed_label": label_number
            }
            supabase.table("image_logs").insert(new_log).execute()
        except Exception as e:
            print(f"Error saving to DB: {e}")

    return label_number

# Keep the old function for backward compatibility
# def predict_dog_breed(image: Image.Image) -> str:
#     if model is None or labels is None:
#         return "Model or labels not loaded."

#     processed_image = preprocess_image(image)
#     preds = model.predict(processed_image)[0]

#     index = int(np.argmax(preds))
#     raw_name = labels.get(str(index), "Unknown")

#     # Clean "n02085620-Chihuahua"
#     if "-" in raw_name:
#         raw_name = raw_name.split("-", 1)[1]

#     clean = raw_name.replace("_", " ").title()
#     return clean
