import matplotlib.pyplot as plt
import seaborn as sns

import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout, InputLayer, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf

import cv2
import os

import numpy as np


img_size = 150
def get_data(data_dir):
    data = [] 
  
        
    for img in os.listdir(data_dir):
        try:
            img_arr = cv2.imread(os.path.join(data_dir, img))[...,::-1] #convert BGR to RGB format
            resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
            data.append(resized_arr)
        except Exception as e:
            print(e)
            
    return np.array(data)

#fetching our train and validation data.
train = get_data("drive/MyDrive/flickr/Images_train")
val = get_data("drive/MyDrive/flickr/Images_test")


#normalisation
x_train = (train) / 255
x_test = (val) / 255

#shape of x_train
print(x_train.shape)
print(x_train[0].shape)

#shape of x_test
print(x_test.shape)
print(x_test[0].shape)

