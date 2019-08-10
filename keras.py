# import required libs

import os
import numpy as np
import pandas as pd
import scipy
import sklearn
import imageio
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D,Convolution2D,MaxPooling2D
from keras.utils import np_utils
from skimage import io
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Preprocess the image, catogorized the labels
nums = []
for each in os.listdir('Train'):
    nums.append(len(os.listdir('Train/'+each)))
age = []
for each in os.listdir('Train'):
    for img in os.listdir('Train/'+each):
        a = img.split('_')
        age.append(a[0])

#Use data generator to create data set used for training, validation, and testing
gen = ImageDataGenerator()

# Training Data Set
train = gen.flow_from_directory('Train/', class_mode='categorical', batch_size=64)

# Validation Data Set
valid = gen.flow_from_directory('Validate/', class_mode='categorical', batch_size=64)

# Test Data Set
test = gen.flow_from_directory('Test/', class_mode='categorical', batch_size=64)

# Build the CNN model
model = Sequential()

# Add the Conv Layers
model.add(Conv2D(64, kernel_size=3, activation='relu',input_shape=(256,256,3)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))

# Add attributes
model.add(Flatten())
model.add(Dense(6, activation='softmax'))

# Compile the model with selected optimizer and loss function
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

# Fit the model with data generator and start traing
model.fit_generator(train, steps_per_epoch=16, validation_data=valid, validation_steps=8,epochs = 3)

# Evaluate the data
loss = model.evaluate_generator(test, steps=24)
print(loss)

# use the model to predict the formatted data
ans = model.predict_generator(test, steps=24)
print(model.summary())

# Save the model
model.save("CNN.h5")