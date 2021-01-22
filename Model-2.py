#Importing the modules we need
import numpy as np
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
from tensorflow import keras
from tensorflow import Tensor
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.initializers import HeNormal
from matplotlib import pyplot as plt
from tensorflow.keras.layers import PReLU, ReLU
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa

#Functions we use to split 5 particle data into EM shower vs non-EM shower/Track particles
def oneHotEncode(inputArray):
   y = inputArray
   b = np.zeros((y.size, y.max()+1))
   b[np.arange(y.size),y] = 1
   return b
 
def particleToEMTrack(inputArray):
   for i in range(inputArray.shape[0]):
       v=inputArray[i]
       if v==0 or v==1:
           inputArray[i] = 0
       else:
           inputArray[i] = 1 
   return inputArray

#Define a data generator
datagen = ImageDataGenerator(width_shift_range=1.0, height_shift_range=1.0)
#Getting and preprocessing the data
X = np.load('X_64_train.napy')
X = X/225.0
y = np.load('y_train_split.npy')
 
#Only used to split data into EM shower vs non EM shower
y = np.argmax(y, axis=1)
y = particleToEMTrack(y)
y = oneHotEncode(y)
 
#Splitting data into train and validation/testing data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

#Load data into data generator
it = datagen.flow(X_train, y_train)

#Define model
input_shape = (64, 64, 1)
initializer = tf.keras.initializers.HeNormal()
model = keras.Sequential(
   [
       keras.Input(shape=input_shape),
      
       layers.Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_initializer=initializer, kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)),
       layers.Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_initializer=initializer, kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)),
       layers.BatchNormalization(),
       layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Dropout(0.1),
 
       layers.BatchNormalization(),
       layers.Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_initializer=initializer, kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)),
       layers.Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_initializer=initializer, kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)),
       layers.BatchNormalization(),
       layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Dropout(0.1),
      
       layers.BatchNormalization(),
       layers.Conv2D(256, kernel_size=(3, 3), activation="relu", kernel_initializer=initializer, kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)),
       layers.Conv2D(256, kernel_size=(3, 3), activation="relu", kernel_initializer=initializer, kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)),
       layers.BatchNormalization(),
       layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Dropout(0.1),
      
       layers.Flatten(),
#         layers.Dropout(0.5),
       layers.Dense(128, activation='relu'),
       layers.BatchNormalization(),
       layers.Dense(5, activation="softmax"),
   ]
)
 
model.summary()

#Compile model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", tfa.metrics.F1Score(num_classes=2)])

#Set directory to store log files for graphing
log_dir = "logs/fit/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#Train model
model.fit(it, batch_size=128, epochs=50, validation_data=(X_val, y_val), verbose=2, callbacks=[tensorboard_callback])
