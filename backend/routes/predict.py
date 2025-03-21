from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import librosa
import tensorflow as tf
import os

router = APIRouter(prefix="")

# Load the model
model_path = os.path.join("models", "speech_emotion_recognition_model.h5")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']

# Extract MFCC Features
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Predict Emotion
def predict_emotion(file_path):
    mfcc_features = extract_mfcc(file_path)
    mfcc_features = np.expand_dims(mfcc_features, axis=0)  # Add batch dimension
    mfcc_features = np.expand_dims(mfcc_features, axis=-1) # Add channel dimension
    
    prediction = model.predict(mfcc_features)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_emotion = emotion_labels[predicted_index]
    return predicted_emotion

# API to receive audio and predict
@router.post("/predict-emotion/")
async def predict_emotion_endpoint(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, "temp.wav")
        
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Predict emotion
        emotion = predict_emotion(file_path)
        return JSONResponse(content={"emotion": emotion})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
