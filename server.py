from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
# import numpy as np
import json

from flask import Flask, request, render_template

app = Flask(__name__)

def eval(filename):

    # Constants
    IM_SHAPE = (200, 200)
    CLASSES = ["Bulbasaur", "Charmander", "Pikachu", "Squirtle"]

    # Load model
    model = load_model("model.h5")

    # Load image
    img = image.load_img(filename, target_size=(IM_SHAPE[0], IM_SHAPE[1]))
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

@app.post("/eval")
def get_predictions():

    # uploaded_file = request.form.get("file")
    uploaded_file = request.files["file"]

    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
        
    # return eval("test_pikachu.jpg")
    return eval(uploaded_file.filename)


@app.get("/eval")
def render_eval_page():
    return render_template("eval.html")

# Start server: "python3 server.py"
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=True)
