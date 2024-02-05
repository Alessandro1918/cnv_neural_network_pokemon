# Load data
# from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the model and use it
from tensorflow.keras.models import load_model
import numpy as np

# Plot some images
import matplotlib.pyplot as plt

# Consts:
DATA_DIR = 'dataset_splited/test'
IM_SHAPE = (200, 200)
SEED = 931
BATCH_SIZE = 4

# Setup data
# Without data augmentation:
#data_generator = ImageDataGenerator(rescale=1./255)
# With data augmentation:
data_generator = ImageDataGenerator(
    # rescale=1./255, 
    rotation_range=20, 
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True, 
    fill_mode='nearest'
)
# Load data
data_generator = data_generator.flow_from_directory(
    DATA_DIR, 
    target_size=IM_SHAPE, 
    shuffle=True, 
    seed=SEED,
    class_mode='categorical', 
    batch_size=BATCH_SIZE
)

classes = list(data_generator.class_indices.keys())
print('Classes: ' + str(classes))   #Classes: ['Bulbasaur', 'Charmander', 'Pikachu', 'Squirtle']

# Load model
model = load_model('model.h5')

# Plot some examples
plt.figure(figsize=(10, 10))
for i in range(9):
    #gera subfigures
    batch = data_generator.next()[0]    # shape (4, 200, 200, 3), floats from 0-255
    item = batch[0]                     # first item of the 4 elements array
    rescaled = item / 255               # shape (200, 200, 3), floats from 0-1
    image = item.astype('uint8')        # shape (200, 200, 3), ints from 0-255    
    reshaped = rescaled.reshape(1, IM_SHAPE[0], IM_SHAPE[1], 3)     # (200, 200, 3) -> (1, 200, 200, 3)
    predictions = model.predict(reshaped)[0]                        # [0, 0.1, 0.6, 0.3]
    print(predictions)
    prediction_class_index = np.argmax(predictions)                 # 2
    prediction_class_name = classes[prediction_class_index]         # "Pikachu"
    title = "{} ({:.2f}%)".format(prediction_class_name, 100 * np.max(predictions))     # "Pikachu (60%)"
    plt.subplot(330 + 1 + i).set_title(title)
    plt.imshow(image)
#plt.show()
plt.savefig('plot_test_' + str(SEED) + '.png')
plt.close()