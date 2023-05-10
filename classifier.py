# Import required libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pickle

# Load the vGG16 model
model = VGG16(weights='imagenet', include_top=True)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Define the target image size
img_size = (224, 224)

# Define a function to preprocess the image
def preprocess_image(image_path):
    # Load the image with the target size
    img = load_img(image_path, target_size=img_size)
    # Convert the image to a numpy array
    img = img_to_array(img)
    # Expand the dimensions of the image to match the input size of the vGG16 model
    img = np.expand_dims(img, axis=0)
    # Preprocess the image for the vGG16 model
    img = preprocess_input(img)
    return img

# Define a function to predict whether an image is a cow or non-cow
def predict_image(image_path):
    # Preprocess the image
    img = preprocess_image(image_path)
    # Use the vGG16 model to predict the image class
    predictions = model.predict(img)
    # Decode the predictions to get the class label
    label = keras.applications.vgg16.decode_predictions(predictions)[0][0][1]
    # Check if the class label is 'cow'
    if label == 'cow':
        return 'Cow'
    else:
        return 'Non-cow'
# Predict the class of an image
image_path = 'tset.jpg'
prediction = predict_image(image_path)
print(prediction)
