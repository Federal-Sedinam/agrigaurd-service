import base64
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_v2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_v3
import numpy as np
import io

# Step 1: Decode the base64 string to an image
def decode_base64_image(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    return image

# Step 2: Preprocess the image
def preprocess_image_for_mobilenet_v2(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = preprocess_input_v2(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Step 2: Preprocess the image
def preprocess_image_for_mobilenet_v3(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = preprocess_input_v3(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def preprocess_image_for_leaf_non_leaf(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = image_array/255
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Step 3: Load the model
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Step 4: Make predictions
def predict(model, preprocessed_image):
    predictions = model.predict(preprocessed_image)
    return predictions