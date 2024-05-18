import json

from fastapi import FastAPI, File, UploadFile
import numpy as np
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
from fastapi.middleware.cors import CORSMiddleware
import requests


#creating an instance of FASTAPI
app = FastAPI()


origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Get the absolute path to the model file
model_path = os.path.abspath(os.path.join("..", "Tomatoes_Plant_Disease_Classification","model_versions","1"))
PREDICTION_MODEL = tf.keras.models.load_model(model_path)
CLASS_NAMES= ['Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_healthy']




#creating a dummy/ckecing endpoint to check server
@app.get("/ping")
async def ping():
    return "The server is up and working!!"


# Reads bytes data as numpy array using PIL's Image and BytesIO
def read_file_as_numpy_array(bytes_data) -> np.ndarray:
    image= np.array(Image.open(BytesIO(bytes_data)))
    return image

# POST endpoint for predicting class and confidence of a plant image
@app.post("/predict")
async def predict(
    file: UploadFile= File(...)  # Expects an uploaded file
):
    image = read_file_as_numpy_array(await file.read())  # Read and convert the uploaded image

    batch_image= np.expand_dims(image,0)  # Prepare image batch for prediction



    prediction_output = PREDICTION_MODEL.predict(batch_image)  # Get prediction from model

    class_of_plant = CLASS_NAMES[np.argmax(prediction_output[0])]  # Get predicted class
    confidence = np.max(prediction_output[0])  # Get confidence score
    return {
        'class': class_of_plant,  # Return predicted class
        'confidence': float(confidence*100)  # Return confidence percentage
    }



#when application runs, it will run uvicorn server on port 8200 and run the application "app" on localhost
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8200)