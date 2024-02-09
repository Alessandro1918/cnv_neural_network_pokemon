import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import json

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
    print(predictions)                                  # [0.0.05347924, 0.0.03757086, 0.0.7415803, 0.0.16736959]
    # class_index = np.argmax(predictions)              # 2
    # class_name = CLASSES[class_index]                 # "Pikachu"

    # Return result as json object
    data = {}
    for i in range(len(CLASSES)):
        data[CLASSES[i]] = float("{:.4f}".format(predictions[i]))
    result = json.dumps(data, indent=2)
    print(result)                                       # {"Bulbasaur": 0.0535, "Charmander": 0.0376, "Pikachu": 0.7416, "Squirtle": 0.1674}
    return result


eval("test_pikachu.jpg")