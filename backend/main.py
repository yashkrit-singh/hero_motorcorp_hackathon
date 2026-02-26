from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
import cv2
import shutil
import os

app = FastAPI()

# Load model once at startup
interpreter = tf.lite.Interpreter(
    model_path="/Users/yashkritsingh/Documents/Hero Mototcorp/edge_training/edge_model.tflite"
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = 224

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def edge_predict(image_path):
    img = preprocess_image(image_path).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    prob = output[0][0]
    return float(prob)

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):

    os.makedirs("uploads", exist_ok=True)
    file_location = f"uploads/{file.filename}"

    # Save uploaded file
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run edge model
    probability = edge_predict(file_location)

    if probability < 0.5:
        return {
            "status": "rejected",
            "reason": "Image quality insufficient",
            "confidence": round(probability, 3)
        }

    return {
        "status": "accepted",
        "confidence": round(probability, 3)
    }