#Load image:
from tensorflow.keras.preprocessing import image
import base64
from PIL import Image
from io import BytesIO
#Predict:
from tensorflow.keras.models import load_model
# import numpy as np
import json
#Serve page to frontend:
import os
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, render_template
from waitress import serve

app = Flask(__name__)

# def eval(filename):       # V2
def eval(file_base64):      # V3

    # Constants
    IM_SHAPE = (200, 200)
    CLASSES = ["Bulbasaur", "Charmander", "Pikachu", "Squirtle"]

    # Load model
    model = load_model("model.h5")

    # Load image
    # img = image.load_img(filename, target_size=(IM_SHAPE[0], IM_SHAPE[1]))        # V2: from file using param = filename
    img = Image.open(BytesIO(base64.b64decode(file_base64))).resize((IM_SHAPE[0], IM_SHAPE[1]))     # V3: from base64 using param = base64 data
    # Convert to array
    img = image.img_to_array(img)                       # shape (200, 200, 3), floats from 0-255
    # Re-scale pixels
    img = img / 255                                     # shape (200, 200, 3), floats from 0-1
    # Re-shape array to model input dimensions
    img = img.reshape(1, IM_SHAPE[0], IM_SHAPE[1], 3)   # (200, 200, 3) -> (1, 200, 200, 3)
    # img = np.expand_dims(img, axis=0)                 # same way to re-shape

    # Predict
    predictions = model.predict(img)[0]
    # print(predictions)                                # [0.0.05347924, 0.0.03757086, 0.0.7415803, 0.0.16736959]
    # class_index = np.argmax(predictions)              # 2
    # class_name = CLASSES[class_index]                 # "Pikachu"

    # Return result as json object
    data = {}
    for i in range(len(CLASSES)):
        data[CLASSES[i]] = float("{:.4f}".format(predictions[i]))
    result = json.dumps(data, indent=2)
    # print(result)                                     # {"Bulbasaur": 0.0535, "Charmander": 0.0376, "Pikachu": 0.7416, "Squirtle": 0.1674}
    return result

# Test function without server:
# print(eval("test_pikachu.jpg"))


# Will be called by frontend's "/eval" page. Get image file from request form, and use it on the "eval" function.
@app.post("/eval")
def get_predictions():

    # uploaded_file = request.form.get("file")
    uploaded_file = request.files["file"]

    # V1: use hardcoded inpup file
    # return eval("test_pikachu.jpg")
    
    # V2: use file from frontend request
    # if uploaded_file.filename != '':
    #     uploaded_file.save(uploaded_file.filename)
    # return eval(uploaded_file.filename)

    # V3: encode / decode image file to base64 and don't save the file on the server:
    if uploaded_file.filename != '':
        file_base64 = base64.b64encode(uploaded_file.read())
        return eval(file_base64)


# Frontend page. Will make a POST request to "/eval"
@app.get("/eval")
def render_eval_page():
    API_URL = os.getenv("API_URL")
    return render_template("eval.html", API_URL=API_URL)

# Start server: "python3 server.py"
if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=4000, debug=True)        #DEV
    serve(app, host="0.0.0.0", port=4000)                   #PROD
