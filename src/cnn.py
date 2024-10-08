#Process data
import os, shutil, splitfolders
#from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#Create CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
#Train
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
#Learning curves
import matplotlib.pyplot as plt
#Eval model
from tensorflow.keras.models import load_model
#Reports
import numpy as np
from sklearn.metrics import confusion_matrix

# Constants:
SEED = 339
BATCH_SIZE = 8
IM_SHAPE = (200, 200)
EPOCHS = 30
PATH_DATASET = os.path.join('src', 'dataset')                       #src/dataset
PATH_DATASET_SPLITED = os.path.join('src', 'dataset_splited')       #src/dataset_splited
PATH_TRAIN = os.path.join(PATH_DATASET_SPLITED, 'train')            #src/dataset_splited/train
PATH_TEST = os.path.join(PATH_DATASET_SPLITED, 'test')
PATH_MODEL = os.path.join('src', 'model.h5') 

# Split Input data into Training, Validation, and Test
shutil.rmtree(PATH_DATASET_SPLITED, ignore_errors=True)             # clear dir, if exists
print('Split data:')
splitfolders.ratio(
    PATH_DATASET, 
    output=PATH_DATASET_SPLITED, 
    seed=SEED , 
    ratio=(.7, 0, .3)
)
# ratio=(.8, 0, .2): Between Train-Test (use validation_split = 0.2 and subset = "training" / "validation")
# ratio=(.6, .2, .2)): Between Train-Val-Test (no need of validation_split, get val_generator data from VAL_DIR='splited/val')

print('Training data:')
#Without data augmentation
#train_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)       # Between Train-Validation
# With data augmentation:
train_generator = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = train_generator.flow_from_directory(
    PATH_TRAIN, 
    target_size=IM_SHAPE, 
    shuffle=True, 
    seed=SEED,
    class_mode='categorical', 
    batch_size=BATCH_SIZE, 
    subset="training"
)

print('Validation data:')
validation_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)
validation_generator = validation_generator.flow_from_directory(
    PATH_TRAIN, 
    target_size=IM_SHAPE, 
    shuffle=False, 
    seed=SEED, 
    class_mode='categorical',
    batch_size=BATCH_SIZE, subset="validation"
)

print('Test data:')
test_generator = ImageDataGenerator(rescale=1./255)
test_generator = test_generator.flow_from_directory(
    PATH_TEST, 
    target_size=IM_SHAPE, 
    shuffle=False, 
    seed=SEED,
    class_mode='categorical', 
    batch_size=BATCH_SIZE
)

classes = list(train_generator.class_indices.keys())
print('Classes: ' + str(classes))

# Define a CNN Model
# More on CNN architecture:
# https://www.datacamp.com/tutorial/convolutional-neural-networks-python
# https://pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/
# https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5
model = Sequential()
#Input layer:
model.add(Conv2D(
    64,                     # n. of filters to learn
    kernel_size=(3, 3),     # shape of the filter
    activation='relu',      # name of the activation_function
    input_shape=(
        IM_SHAPE[0],        # rows
        IM_SHAPE[1],        # collumns
        3                   # channels
    )
))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
#Hidden layer:
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
#Output layer:
model.add(Dense(len(classes), activation='softmax'))

# Print details about each layer of the model
model.summary()

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

#Callback to save the best model
callbacks_list = [
    ModelCheckpoint(
        filepath=PATH_MODEL,
        monitor='accuracy',     #options: "accuracy", "val_loss" 
        save_best_only=True, 
        verbose=1
    ),
    EarlyStopping(
        monitor='accuracy', 
        patience=10, 
        verbose=1
    )
]

#Training
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks_list,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    verbose=0
)

#Learning Curves
history_dict = history.history
plt.figure(figsize=(10, 10))

#Loss
#plt.subplot(2,1,1)
#loss_values = history_dict['loss']
#val_loss_values = history_dict['val_loss']
#axis_x = range(1, len(loss_values) + 1)
#plt.plot(axis_x, loss_values, 'bo', label='Training loss')
#plt.plot(axis_x, val_loss_values, 'b', label='Validation loss')
#plt.title('Training and validation Loss and Accuracy')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()

#Accuracy
#plt.subplot(2,1,2)
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
axis_x = range(1, len(acc_values) + 1)
plt.plot(axis_x, acc_values, color='blue', label='Training acc')
plt.plot(axis_x, val_acc_values, color='red', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

#Show
#plt.show()
#Save
#filename = sys.argv[0].split('/')[-1]
plt.savefig('plot_learning.png')
plt.close()

#Evaluating the model
print('\nEvaluating the model:')

# Load the best saved model
model = load_model(PATH_MODEL)

# Training dataset
#score = model.evaluate_generator(...) # Deprecated
score = model.evaluate(train_generator)
print('Training loss:', score[0])
print('Training accuracy:', score[1])

# Validation dataset
score = model.evaluate(validation_generator)
print('Val loss:', score[0])
print('Val accuracy:', score[1])

# Test dataset
score = model.evaluate(test_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Reports

#Confusion Matrix
#Y_pred = model.predict_generator(test_generator)   # Deprecated
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix:')
print(confusion_matrix(test_generator.classes, y_pred))