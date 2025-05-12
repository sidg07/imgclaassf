import io
from typing import List, Dict

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = FastAPI()

# Add CORS middleware to allow cross-origin requests (replace with your frontend origin)
app.add_middleware(
    CORSMiddleware,
    # Or "*" for any origin, but be careful in production
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained Keras model.  Make sure the path is correct.
try:
    # Replace with your model's path
    model = tf.keras.models.load_model("my_model.h5")
    # print(model.summary()) #Uncomment this line to see the structure of the model
except Exception as e:
    print(f"Error loading model: {e}")
    # Consider raising an exception here to prevent the app from running with an invalid model.
    # Set model to None to prevent further errors, and check for None before predictions.
    model = None


# Define the expected image dimensions and labels, adapt to your needs
IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNELS = 3
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']  # Example: CIFAR-10


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocesses the image data for use with the TensorFlow model.

    Args:
        image_bytes (bytes): The raw bytes of the image.

    Returns:
        np.ndarray: A NumPy array representing the preprocessed image,
                      ready for the model to make a prediction.  Will return
                      None if there is an error.
    """
    try:
        # Decode the image bytes using PIL (supports various formats).
        image = Image.open(io.BytesIO(image_bytes))
        # Resize the image to the target dimensions.
        image = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
        # Convert the image to a NumPy array.
        image_array = np.array(image)
        # Normalize the pixel values to be between 0 and 1.
        image_array = image_array / 255.0
        # Expand the dimensions of the array to create a batch of size 1,
        # as the model expects input in batch form.
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None  # Important: Return None on error


def predict(image_array: np.ndarray) -> Dict:
    """
    Makes a prediction using the loaded machine learning model.

    Args:
        image_array (np.ndarray): The preprocessed image data.

    Returns:
        dict: A dictionary containing the prediction:
            {
                "label": str,     # The predicted class name
                "confidence": float # The confidence of the prediction (probability)
            }
            Returns an empty dict on error.
    """
    if model is None:
        print("Model not loaded, cannot make prediction.")
        return {}

    try:
        # Make a prediction using the model.
        predictions = model.predict(image_array)
        # Get the predicted class index.
        predicted_class_index = np.argmax(predictions[0])
        # Get the predicted class name.
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        # Get the confidence of the prediction.  Convert to a float.
        confidence = float(predictions[0][predicted_class_index])

        # Return the result as a dictionary.
        return {
            "label": predicted_class_name,
            "confidence": confidence,
        }
    except Exception as e:
        print(f"Error making prediction: {e}")
        return {}  # Return empty dict on error


@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)) -> Dict:
    """
    Endpoint for classifying an image.

    Args:
        file (UploadFile): The image file to classify, sent in the request.

    Returns:
        dict: A dictionary containing the classification result.  Returns a 400
              error if the model failed to load, the image could not be processed,
              or the prediction failed.
    """
    if model is None:
        raise HTTPException(status_code=400, detail="Model not loaded.")

    # Read the image file as bytes.
    image_bytes = await file.read()

    # Preprocess the image.
    image_array = preprocess_image(image_bytes)
    if image_array is None:
        raise HTTPException(status_code=400, detail="Invalid image data.")

    # Make a prediction.
    prediction = predict(image_array)
    if not prediction:
        raise HTTPException(
            status_code=400, detail="Failed to classify image.")

    # Return the prediction as JSON.
    print(prediction)
    return prediction


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
