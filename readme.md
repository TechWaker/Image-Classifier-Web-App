# Glass or Table Classifier Web App
Earlier, I created a model which uses ResNet50 as its first layer and a dense layer with 2 nodes as the output layer. This new model was then trained to distinguish between images of glass and tables. The dataset used is Glasses vs Tables by [Muhammed Buyukkinaci](https://github.com/MuhammedBuyukkinaci).

## Deploying a Keras Model
I create a Keras model with the same architecture (ResNet50 with pretrained weights as first layer and a Dense layer with 2 nodes as output). I load the saved weights. Then, I use the image preprocessing from Keras to create a function that will prepare incoming images to be fed into our trained model. Since I am loading pre-trained weights, I do not need to perform any training again.

## Flask App
If the request method to the homepage is GET, we display a file upload form. When an image file is uploaded, it is saved to the `static` folder, is prepared to be fed into our model to make a prediction and ultimately a prediction is made using `model.predict`. This value is returned and displayed on the web page along with the uploaded image.

## How to Use This App
Clone or download this repo. You will also need these packages: TensorFlow, Keras, Pillow and Flask. Then, run the app with `flask run`. You may need to set environment variable `FLASK_APP` to `app.py`.

Since the model was trained on images of glass or tables, you should ideally upload images of either glass or table and hopefully, the model will be able to predict accurately.
