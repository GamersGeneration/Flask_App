#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      rePUBLIC
#
# Created:     02-04-2023
# Copyright:   (c) rePUBLIC 2023
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np

# Load the model from disk
model = tf.keras.models.load_model('cow_detector.h5')

# Define a function for predicting whether an image contains a cow or not
def predict_cow(image):
    # Preprocess the image data
    image = tf.io.decode_image(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    image = image / 255.0

    # Make a prediction
    prediction = model.predict(np.array([image]))

    # Return the prediction as a string
    if prediction[0][0] > 0.5:
        return 'cow'
    else:
        return 'not a cow'
