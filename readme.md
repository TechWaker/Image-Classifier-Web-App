# Glass or Table Classifier Web App
This example takes a model created in one of our earlier [Transfer Learning](https://github.com/am1tyadav/Computer-Vision-with-Convolutional-Neural-Networks/blob/master/02%20-%20Transfer%20Learning.ipynb) experiments. We created a model which uses ResNet50 as its first layer and a dense layer with 2 nodes as the output layer. This new model was then trained to distinguish between images of glass and tables. The dataset used is Glasses vs Tables by [Muhammed Buyukkinaci](https://github.com/MuhammedBuyukkinaci).

## Deploying a Keras Model
Much of the code remained exactly the same as the Jupyter Notebook linked above. We create a Keras model with the same architecture (ResNet50 with pretrained weights as first layer and a Dense layer with 2 nodes as output). We load the saved weights. Just like in the notebook, we use the image preprocessing from Keras to create a function that will prepare incoming images to be fed into our trained model. Since we are loading pre-trained weights, we do not need to perform any training again.

## Flask App
If the request method to the homepage is GET, we display a file upload form. When an image file is uploaded, it is saved to the `static` folder, is prepared to be fed into our model to make a prediction and ultimately a prediction is made using `model.predict`. This value is returned and displayed on the web page along with the uploaded image.

## How to Use This App
Clone or download this repo. In the app's folder, create a virtual environment with `virtualenv`, activate the environment. On Mac, this would be `source venv/bin/activate` and on Windows, this would be `venv/scripts/activate`. Once the environment is activated, you will need to install a few packages: TensorFlow, Keras, Pillow, and Flask. Create an empty folder called `static` where the uploaded image files will be kept. Then, run the app with `flask run`.

Since the model was trained on images of glass or tables, you should ideally upload images of either glass or table and hopefully, the model will be able to predict accurately.
