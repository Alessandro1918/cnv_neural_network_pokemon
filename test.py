# Load data
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Plot some images
import matplotlib.pyplot as plt

# Load the model and use it
from tensorflow.keras.models import load_model
import numpy as np


DATA_DIR = 'dataset_splited/test'
IM_SHAPE = (200, 200)
SEED = 931
BATCH_SIZE = 4


#Load data
#Without data augmentation
#data_generator = ImageDataGenerator(rescale=1./255)
# With data augmentation:
data_generator = ImageDataGenerator(rescale=1./255, 
                                     rotation_range=20, width_shift_range=0.2, 
                                     height_shift_range=0.2, shear_range=0.2, 
                                     zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
data_generator = data_generator.flow_from_directory(
    DATA_DIR, 
    target_size=IM_SHAPE, 
    shuffle=True, 
    seed=SEED,
    class_mode='categorical', 
    batch_size=BATCH_SIZE
)

classes = list(data_generator.class_indices.keys())
print('Classes: ' + str(classes))   #Classes: ['Bulbassaur', 'Charmander', 'Pikachu', 'Squirtle']


#Load model
model = load_model('model.h5')


# Visualizing some examples
plt.figure(figsize=(10, 10))
for i in range(9):
    #gera subfigures
    batch = data_generator.next()[0]*255
    image = batch[0].astype('uint8')
    reshaped = image.reshape(1, IM_SHAPE[0], IM_SHAPE[1], 3)    # This block uses the model
    prediction = model.predict(reshaped)                        # to predict the image classes
    prediction_class = np.argmax(prediction, axis=-1)           # [0], [3], [2], ...
    #plt.subplot(330 + 1 + i)
    plt.subplot(330 + 1 + i).set_title(classes[prediction_class[0]])
    plt.imshow(image)
#plt.show()
plt.savefig('plot_test_' + str(SEED) + '.png')
plt.close()