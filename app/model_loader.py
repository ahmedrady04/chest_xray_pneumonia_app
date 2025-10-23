from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from pathlib import Path

# === Configuration ===
# Correct path: go up one directory, then into /models/
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.h5"
print(f"🔄 Loading model from: {MODEL_PATH}")
IMG_SIZE = (224, 224)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# === Load model once ===
print("🔄 Loading model from:", MODEL_PATH)
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

def preprocess_image(img_path):
    """Load and preprocess the image for CNN prediction."""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_image(img_path):
    """Run prediction on a single image path."""
    img_tensor = preprocess_image(img_path)
    preds = model.predict(img_tensor)
    predicted_class = CLASS_NAMES[np.argmax(preds)]
    confidence = float(np.max(preds))
    return predicted_class, confidence
