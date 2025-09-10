import os
import json
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

MODEL_PATH = os.path.join("models","keras_models", "model-mobilenet-RMSprop0.0002-008-0.995584-0.711503.h5")
                                              
# Load your trained model
model = load_model(MODEL_PATH)
print("Model loaded successfully !! Check http://127.0.0.1:5000/")

with open(os.path.join("static","food_list", "food_list.json"), "r", encoding="utf8") as f:
    food_labels = json.load(f)
class_names = sorted(food_labels.keys())
label_dict = dict(zip(range(len(class_names)), class_names))

food_calories = pd.read_csv(os.path.join("static","food_list", "Food_calories.csv"))

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    # Preprocessing the image
    x = image.img_to_array(img) / 255
    x = np.expand_dims(x, axis=0)
    return x

@app.route("/", methods=["GET"])
def Home():
    # Main page
    return render_template('Meal_Assessment_System.html')

@app.route("/predict", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        # Get the file from post request
        f = request.files["image"]

        # Save the file to ./upload_image
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, "upload_image", secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        image = prepare_image(file_path)
        preds = model.predict(image)
        predictions = preds.argmax(axis=-1)[0]
        pred_label = label_dict[predictions]

        food_retrieve = food_calories[food_calories["name"]==pred_label]

        food_nutrional_min = food_retrieve["nutritional value min,kcal"]
        food_nutrional_min = np.array(food_nutrional_min)
        food_nutrional_min = str(food_nutrional_min)

        food_nutrional_max = food_retrieve["nutritional value max,kcal"]
        food_nutrional_max = np.array(food_nutrional_max)
        food_nutrional_max = str(food_nutrional_max)

        Unit = food_retrieve["unit"]
        Unit = np.array(Unit)
        Unit = str(Unit)

        Calories = food_retrieve["average cal"]
        Calories = np.array(Calories)
        Calories = str(Calories)

        return (
            "<center><i><h4>" + pred_label.title() + "</h4></i> "
            + "<b><h3>Probability</h3></b><h4>" + str(preds.max(axis=-1)[0]) + "</h4><br><br>"
            + "<div class=\"heading-section\"><h2 class=\"mb-4\"><span>Nutritional Facts</span></h2></div><hr>"
            + "<h5><b>Nutritional Value - Min (kcal) &nbsp;: &nbsp;</b>" + food_nutrional_min + "<br><br>"
            + "<b>Nutritional Value - Max (kcal) &nbsp;: &nbsp;</b>" + food_nutrional_max + "<br><br>"
            + "<b>Avg Calories &nbsp;: &nbsp;</b>" + Calories + "<br><br>"
            + "<b>Unit &nbsp;: &nbsp;</b>" + Unit + "</h5><br><br>"
        )

    return None

if __name__ == "__main__":
    # Serve the app with gevent
    http_server = WSGIServer(("0.0.0.0", 5000), app)
    http_server.serve_forever()