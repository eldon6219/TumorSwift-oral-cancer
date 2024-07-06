import numpy as np
from flask import Flask , render_template , request
from keras.preprocessing import image
from keras.applications.efficientnet import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# from keras.applications.vgg16 import preprocess_input
# from keras.applications.vgg16 import decode_predictions
# from keras.applications.vgg16 import VGG16
import tensorflow as tf
import requests
import joblib


url = "https://github.com/eldon6219/TumorSwift-oral-cancer/blob/main/Oral.pkl?raw=true"
response = requests.get(url, stream=True)

with open("Oral.pkl", "wb") as f:
    for chunk in response.iter_content(1024):
        f.write(chunk)

model = joblib.load("Oral.pkl")
class_labels = ['Normal', 'Positive Oral Cancer']


app = Flask(__name__) 

@app.route('/' , methods=['GET'])
def hello_world():
    return render_template("index.html")

@app.route('/predict' , methods=['POST'])
def predict():

    # Get the values from the form in the request
    imagefile=request.files['imagefile']
    image_path = "./img/" + imagefile.filename
    imagefile.save(image_path)

    # img1 = image.load_img(image_path, target_size=(224, 224,3))
    # img_array = image.img_to_array(img1)
    image = load_img(image_path , target_size=(224,224,3))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    # image = image.reshape((1 ,image.shape[0] , image.shape[1] , image.shape[2]))
    # image = preprocess_input(image)
    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_labels[predicted_class_index]

    # predictions = model.predict(image)
    # result = decode_predictions(predictions)
    # result = result[0][0]
    return render_template("result.html" , predictions=predicted_class_label)

if __name__ == '__main__':
    app.run()
