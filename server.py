#   cURL
# 	curl -X POST -F image=@fish_type.jpg 'http://localhost:5000/predict'

#   Imports
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io

#   Init app, model & set fish species
app = flask.Flask(__name__)
model = None
fish_species = [
    "Black Sea Sprat" ,
    "Gilt-Head Bream",
    "Hourse Mackerel" ,
    "Red Mullet" ,
    "Red Sea Bream" ,
    "Sea Bass",
    "Shrimp" ,
    "Striped Red Mullet",
    "Trout" 
]


#   Check for allowed extensions
allowed_extensions = set(['png', 'jpg', 'jpeg', 'gif'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


#   Load saved Keras Model
def load_model():
    global model
    model = keras.models.load_model("model")


#   Preprocessing received image to predict
def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")

    img_array = []
    image = image.resize(target)
    image = img_to_array(image)
    img_array.append(image)
    img_array = np.array(img_array)
    img_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)
    data = img_datagen.flow(img_array,batch_size=32,seed=42)
    return data


#   Main post function
@app.route("/predict", methods=["POST"])
def predict():
    #   Dictionary for return
    data = {"success": False,"message": "No files found","predictions": []}
    status_code = 400
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            data["message"] = "File type not supported"
            if allowed_file(flask.request.files.get("image").filename):
                try:
                    #   Read the image in PIL format
                    image = flask.request.files["image"].read()
                    image = Image.open(io.BytesIO(image))

                    #   Preprocessing from function
                    image = prepare_image(image, target=(224, 224))

                    #   Predict using loaded model
                    preds = model.predict(image)
                    i = 0

                    #   Loop through probability array
                    for prob in preds[0]:
                        r = {"label": fish_species[i], "probability": float(prob)}
                        data["predictions"].append(r)
                        i+=1
                    
                    #   Set success status
                    status_code = 200
                    data["message"] = "Successful"
                    data["success"] = True
                except Exception:
                    data["message"] = "Internal Error"
                    status_code = 500
            
    # return the data dictionary as a JSON response
    data_json = flask.jsonify(data)
    data_json.status_code = status_code
    return data_json


#   Load model and start server
if __name__ == "__main__":
	load_model()
	app.run(debug=False)