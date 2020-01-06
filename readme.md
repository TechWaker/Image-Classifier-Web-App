# Image Classifier Web App

This image classifier web app can serve as a template for anyone looking to deploy a `tf.keras` model using Flask.

The model used in this example was then trained to distinguish between images of glass and tables. The dataset used is Glasses vs Tables by [Muhammed Buyukkinaci](https://github.com/MuhammedBuyukkinaci).

To use this app, clone or download this repo. Run the app with `flask run`. You may need to set environment variable `FLASK_APP` to `app.py`. Since the model was trained on images of glass or tables, you should ideally upload images of either glass or table and hopefully, the model will be able to predict accurately.