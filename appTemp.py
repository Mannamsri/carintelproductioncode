from flask import Flask, render_template, request, session, redirect, url_for, session
import requests
from flask_wtf import FlaskForm, form
from wtforms import TextField, SubmitField, FileField
from wtforms.validators import NumberRange
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import PIL

import joblib


def return_prediction(model, cimage):
    image_shape = (442, 639, 3)
    input_image = image.load_img(cimage, target_size=image_shape)
    print('input_image')
    input_image1 = image.img_to_array(input_image)
    input_image2 = np.expand_dims(input_image1, axis=0)
    classes = np.array(['DAMAGED', 'WHOLE'])
    class_ind = model.predict_classes(input_image2)
    return classes[class_ind][0]


app = Flask(__name__)
# Configuring a secret SECRET_KEY
app.config['SECRET_KEY'] = 'mysecretkey'

UPLOAD_FOLDER = "C:/Users/Thaswika/Desktop/ICanCode/CarintelProductionCode/static/image/"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Loading trained Model
car_model = load_model("FindingDamagedCar.h5")


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        global item
        global imageresults
        global absfilepath
        if request.files:
            # save the file to temp folder
            item = request.files["image"]
            filename = item.filename
            # tempfilename = filename.split(".")
            actualfilename = "OutputImage.jpg"
            absfilepath = os.path.join(app.config['UPLOAD_FOLDER'], actualfilename)
            item.save(absfilepath)
            image_shape = (442, 639, 3)
            input_image = image.load_img(absfilepath, target_size=image_shape)
            print('input_image')
            input_image1 = image.img_to_array(input_image)
            input_image2 = np.expand_dims(input_image1, axis=0)
            classes = np.array(['DAMAGED', 'WHOLE'])
            class_ind = car_model.predict_classes(input_image2)
            print('Uploaded Image has been processed by Deep learning model successfully and the result is')
            imageresults = classes[class_ind][0][0]
            print(imageresults)
        return redirect(url_for('prediction'))
    return render_template("home.html")


@app.route('/prediction')
def prediction():
    print(imageresults)
    return render_template('prediction.html', results=imageresults, absfilepath=absfilepath)


if __name__ == '__main__':
    app.run(debug=True)
