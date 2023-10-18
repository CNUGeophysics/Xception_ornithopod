#install library

!pip install tensorflow-addons==0.16.1

import tensorflow as tf
import os
import glob
import shutil
import matplotlib.pyplot as plt
import random 
import numpy as np 
import cv2 
import tensorflow_addons as tfa

from keras import regularizers
from google.colab import drive
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix,f1_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.applications import MobileNet, Xception, ResNet50, InceptionV3, NASNetLarge, VGG16, VGG19,ResNet152V2
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#define direction
root_dir = '/content/drive/MyDrive'

src_root_dir = os.path.join(root_dir, 'large_ornithopod/train') 
dst_root_dir = os.path.join(root_dir, 'large_ornithopod/test/')

label_name_list = os.listdir(src_root_dir)

print(label_name_list)

for label_name in label_name_list:
  dst_label_name_dir = dst_root_dir + label_name 

#define imagegenerator and image directory 

img_width = 331
img_height = 331

train_dir = os.path.join(root_dir, 'large_ornithopod/train')
validation_dir = os.path.join(root_dir, 'large_ornithopod/train')
test_dir = os.path.join(root_dir, 'large_ornithopod/test/')

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.1,
                                   height_shift_range=0.1, zoom_range=0.2,
                                   validation_split = 0.15, fill_mode = 'constant',cval = 0.)

validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)

train_generator = train_datagen.flow_from_directory(train_dir, batch_size=16, color_mode = 'rgb',
                                                    class_mode= 'sparse', subset = 'training',
                                                    target_size = (img_width, img_height)) #train set

validation_generator = validation_datagen.flow_from_directory(validation_dir, batch_size=16, color_mode = 'rgb',
                                                    class_mode= 'sparse', subset = 'validation',
                                                    target_size = (img_width, img_height)) #validation set 

print(train_generator.class_indices)

#define neural network structure 
from keras.layers import BatchNormalization

base_model = Xception(weights = None, include_top= False, input_shape=(img_width, img_height,3))

model = Sequential()

model.add(base_model)
model.add(Flatten())
model.add(Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(32, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(6, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-6), metrics=['accuracy',tfa.metrics.F1Score(num_classes=3, average= 'macro')])

#view model structure 
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png')
plot_model(model, to_file='model_shapes.png', show_shapes=True)
model.summary()


#training
earlyStopping = EarlyStopping(monitor='val_loss', patience = 100)
hist = model.fit(train_generator, validation_data = validation_generator, epochs=1000, verbose=1, callbacks=[earlyStopping])

#view training result 
plt.figure('1', figsize=(50,8))
plt.title('loss trend',fontsize = 50)
plt.grid()
plt.xlabel('epochs',fontsize = 30)
plt.ylabel('loss',fontsize = 30)
plt.plot(hist.history['loss'], label='train', color = 'red',linewidth = 4)
plt.plot(hist.history['val_loss'], '--',label='validation', color = 'navy',linewidth = 4)
plt.xticks(fontsize = 40)
plt.yticks(fontsize = 40)
plt.legend(loc='best',fontsize = 50)
plt.show()

plt.figure('2', figsize=(50,8))
plt.title('accuracy trend',fontsize = 50)
plt.grid()
plt.xlabel('epochs',fontsize = 30)
plt.ylabel('accuracy',fontsize = 30)
plt.plot(hist.history['accuracy'], label='train', color = 'red',linewidth = 4)
plt.plot(hist.history['val_accuracy'], '--', label='validation',color = 'navy',linewidth = 4)
plt.xticks(fontsize = 40)
plt.yticks(fontsize = 40)
plt.legend(loc='best',fontsize = 50)
plt.show()

#model save
model.save('xception_test.h5')
