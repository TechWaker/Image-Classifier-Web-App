import tensorflow as tf
import keras
import os
import numpy as np

from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

UPLOAD_FOLDER = 'static/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
class_names = ['Glass', 'Table']

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

print('Creating Model..')

model = Sequential([
    ResNet50(include_top = False, pooling = 'avg', weights = 'imagenet'),
    Dense(2, activation = 'softmax')
])

# Do NOT to train first layer (ResNet) model. It is already trained
model.layers[0].trainable = False

# Compile the model as well
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

print('Model created..')
print('Loading model weights..')

model.load_weights('model/gvt_model.h5')

print('Weights loaded..')

global graph
graph = tf.get_default_graph()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_images(image_paths):
    images = [load_img(image_path, target_size = (224, 224)) for image_path in image_paths]
    images_array = np.array([img_to_array(image) for image in images])
    return preprocess_input(images_array)

def make_prediction(filename):

    print('Filename is ', filename)
    fullpath = 'static/{}'.format(filename)

    test_data = prepare_images([fullpath])

    with graph.as_default():
        predictions = model.predict(test_data)

    print(predictions[0])

    return class_names[np.argmax(predictions[0])]

def hello(filename):

    return make_prediction(filename)

# ROUTES
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            output = hello(filename)
            path_to_image = url_for('static', filename = filename)

            return '''
            <!doctype html>
            <title>Glass or Table Predictor</title>
            <h1>Prediction: ''' + output + '''</h1>
            <h3><a href = "/">Go back</a></h3>
            <div><img src="''' + path_to_image + '''"></div>
            '''
    return '''
    <!doctype html>
    <title>Glass or Table Predictor</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''