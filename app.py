#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      rePUBLIC
#
# Created:     01-04-2023
# Copyright:   (c) rePUBLIC 2023
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from flask import redirect,url_for, send_from_directory
import os
#import pickle
from flask import Flask, render_template

app = Flask(__name__, template_folder='templates', static_folder='static')
import io
import random
#import joblib
#import tensorflow
from flask import Flask, render_template, request
import classifier

from werkzeug.utils import secure_filename

from PIL import Image
from torchvision import models, transforms
import torch
#import cv2
#import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash
import base64
#from PIL import Image
#import numpy as np
from flask import Flask, render_template, request, jsonify


#model = joblib.load('path/to/saved/model.pkl')
@app.route('/')
@app.route('/home')
def home():
    #return 'Hello, World!'

    #if 'logged_in' in session and session['logged_in']:
    return render_template('index.html')
    #else:
        #return redirect(url_for('login'))
app.secret_key = 'lifeisgood'







# Define a route to handle predictions

"""from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
# Create instance of LoginManager
login_manager = LoginManager(app)

# Create a User class that inherits from UserMixin
class User(UserMixin):
    pass

# Set up a user_loader function to load a user from a user_id
@login_manager.user_loader
def load_user(user_id):
    # In this example, we're just using a single user with the username 'admin' and password 'password'
    if user_id == 'admin':
        user = User()
        user.id = user_id
        return user
    return None

# Add a login route to authenticate users
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'password':
            user = User()
            user.id = username
            login_user(user)
            session['logged_in'] = True
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password')


    # Choose a random image URL from the list
    return render_template('login.html')

# Add a logout route to log out users
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Add a current_user route to get the current user
@app.route('/current_user')
def current_user():
    if current_user.is_authenticated:
        return jsonify({'username': current_user.id})
    else:
        return jsonify({'username': 'anonymous'})"""
@app.route('/my-predict')
def predict():
    file_name = 'key.txt'  # Replace with your file name

    try:
        with open(file_name, 'r') as file:
            file_contents = file.read()
            return render_template('predict.html',file_contents=file_contents)
    except IOError as e:
        return f"Error reading file: {e}"

@app.route('/predict-damage')
def damage():
    file_name='key.txt'
    try:
        with open(file_name, 'r') as file:
            file_contents = file.read()
            return render_template('damage.html',file_contents=file_contents)
    except IOError as e:
        return f"Error reading file: {e}"

model = models.resnet50(pretrained=True)
model.eval()
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


@app.route('/predict', methods=['POST'])
def output():
    # Load and preprocess the input image
    image_file = request.files['image']
    image = Image.open(image_file)
    input_tensor = preprocess_image(image)

    # Perform prediction using the pre-trained model
    with torch.no_grad():
        output = model(input_tensor)

    # Get the predicted class labels
    _, predicted_idx = torch.max(output, 1)
    predicted_label = str(predicted_idx.item())

    # Return the prediction result as a JSON response
    result = {'prediction': predicted_label}
    return jsonify(result)
#@app.route('/course')
#@app.route('/about')
#@app.route('/contact')
#@app.route('/blog')
#def serve_static():
#    root_dir = os.path.dirname(os.getcwd())
#    return send_from_directory(os.path.join(root_dir, 'static'), 'course.html')
if __name__ == '__main__':
    app.run(debug=True)