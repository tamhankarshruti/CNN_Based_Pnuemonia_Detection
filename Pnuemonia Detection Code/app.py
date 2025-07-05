import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

# Load pre-trained model
base_model = VGG19(include_top=False, input_shape=(128, 128, 3))
x = base_model.output
flat = Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)

#Fully connected layers for binary classification
model_03 = Model(base_model.inputs, output)
model_03.load_weights('model_weights/vgg_unfrozen.h5')

#Initialization of Flask App
app = Flask(__name__)
print('Model loaded. Check http://127.0.0.1:5000/')


# Function to return class name, maps prediction with binary classification
def get_className(classNo):
    if classNo == 0:
        return "Normal"
    elif classNo == 1:
        return "Pneumonia"


# Function to detect if image is an X-ray
def is_xray_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #measures how different the color channel is from the grayscale version.
    if np.std(image[..., 0] - gray_image) < 10 and np.std(image[..., 1] - gray_image) < 10 and np.std(image[..., 2] - gray_image) < 10:
        #to check the img is not too bright or not too dark
        avg_brightness = np.mean(gray_image)
        if 40 < avg_brightness < 200:
            return True
    return False


# Main prediction logic
def getResult(img):
    image = cv2.imread(img)  # Reads the image
    if not is_xray_image(image):
        return "Please upload xray image"

    image = Image.fromarray(image, 'RGB')
    image = image.resize((128, 128))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0) # (1,128,128,3) Batch Size ip
    result = model_03.predict(input_img)
    #find the index with the highest probability
    result01 = np.argmax(result, axis=1)
    return get_className(result01[0])


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        result = getResult(file_path)
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
