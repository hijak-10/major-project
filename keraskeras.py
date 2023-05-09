import tensorflow as tf 
from tensorflow.python import keras
import matplotlib.pyplot as plt
import cv2
import os 
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
import keras.utils as image
from PIL import Image
import scipy.integrate as integrate 

train = ImageDataGenerator(rescale=1/255)
train_dataset= train.flow_from_directory(r"C:\Users\lENOVOO\Desktop\Mproject\TT\Data",target_size=(400,400),batch_size=200,class_mode='categorical')

validation_dataset= train.flow_from_directory(r"C:\Users\lENOVOO\Desktop\Mproject\TT\Vdata",target_size=(400,400),batch_size=200,class_mode='categorical')

test_dataset= train.flow_from_directory(r"C:\Users\lENOVOO\Desktop\Mproject\TT\Tdata",target_size=(400,400),batch_size=200,class_mode='categorical')

model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(400,400,3)),
                                  tf.keras.layers.MaxPool2D(2,2),
                                  #
                                  tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
                                  tf.keras.layers.MaxPool2D(2,2),
                                  #
                                  tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                  tf.keras.layers.MaxPool2D(2,2),
                                  ##
                                  tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
                                  tf.keras.layers.MaxPool2D(2,2),
                                  ##
                                  tf.keras.layers.Flatten(),
                                  ##
                                  tf.keras.layers.Dense(64,activation='relu'),
                                  ##
                                  tf.keras.layers.Dense(1,activation='sigmoid')
                                  ])

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop( learning_rate=0.0001),
              metrics=['accuracy'])

logsdir='logs'
tensorboard_callback= tf.keras.callbacks.TensorBoard(log_dir=logsdir)
hist = model.fit(train_dataset,epochs=5,steps_per_epoch=10,validation_data= validation_dataset,callbacks=[tensorboard_callback])