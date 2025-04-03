from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware  # ✅ Import CORS Middleware
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import uvicorn
import io
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv(dotenv_path=".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
# Load model and label encoder
MODEL_PATH = "leaf_classifier.h5"
LABELS_PATH = "labels.npy"

model = load_model(MODEL_PATH)
label_classes = np.load(LABELS_PATH)

# Configure Gemini API

genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")

app = FastAPI()

# ✅ Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["http://localhost:3000"] or your Expo URL for security
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers (Content-Type, Authorization, etc.)
)

# Function to preprocess image
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Function to fetch medicinal properties
def get_medicinal_properties(leaf_name):
    prompt = f"""
    Provide medicinal properties and diseases cured by {leaf_name} in brief bullet points:
    - Three key medicinal benefits
    - Three diseases or ailments it helps to cure
    """
    response = model_gemini.generate_content(prompt)
    return response.text if response else "Information not available"

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file:
        return {"error": "No file received"}
    image_bytes = await file.read()
    image = preprocess_image(image_bytes)
    predictions = model.predict(image)
    predicted_class = label_classes[np.argmax(predictions)]
    confidence = np.max(predictions)
    
    # Get medicinal properties
    medicinal_info = get_medicinal_properties(predicted_class)
    
    return {
        "leaf_name": predicted_class,
        "confidence": float(confidence),
        "medicinal_properties": medicinal_info
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
