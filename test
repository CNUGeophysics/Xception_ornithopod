#define library 
!pip install tensorflow-addons==0.16.1
import tensorflow_addons as tfa
import random 
import os 
import numpy as np 
import cv2 
import glob

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix,f1_score
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from numpy import argmax
from keras.models import load_model

#define model directory 
model = load_model('/content/drive/MyDrive/xception_test.h5', custom_objects={"f1_score": f1_score})
root_dir = '/content/drive/MyDrive'
img_width = 331  
img_height = 331

#difine label and other directory 
label_dict = {'caririchnium': 0, 'hadrosauropodus': 1, 'iguanodontipus': 2}
test_image_files_list = glob.glob(root_dir+ '/large_ornithopod/test_image_files/*.jpg')
random.shuffle(test_image_files_list)
test_num = 12
test_image_files = test_image_files_list[:test_num]
label_list = []


for i in range(len(test_image_files)):
  label = test_image_files[i].split('/')[-1].split('.')[0].strip()
  label_list.append(label_dict[label])

src_img_list = []

for i in range(len(test_image_files)):
  src_img = cv2.imread(test_image_files[i], cv2.IMREAD_COLOR)
  src_img = cv2.resize(src_img, dsize=(img_width, img_height))
  src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
  src_img = src_img/ 255.0

  src_img_list.append(src_img)

src_img_array = np.array(src_img_list)
label_array = np.array(label_list)

print(label_list)

#prediction test image 
pred = model.predict(src_img_array)
print(pred)
print ('prediction shape', pred.shape)

import matplotlib.pyplot as plt
class_names = ['caririchnium', 'hadrosauropodus', 'iguanodontipus']

plt.figure('figure1',figsize=(20,20))

for pos in range(len(pred)):

  plt.subplot(5,3,pos+1)
  
  plt.axis('OFF')

  label_str = class_names[label_array[pos]]
  pred_str = class_names[np.argmax(pred[pos])]

  plt.title( 'Caririchnium : '+str("{:.2f}".format(pred[pos][0]*100))+'%'+'\n Hadrosauropodus : '+str("{:.2f}".format(pred[pos][1]*100
  ))+'%''\n Iguanodontipus : '+str("{:.2f}".format(pred[pos][2]*100))+'%', fontsize = 20)

  plt.imshow(src_img_array[pos])

plt.tight_layout()
plt.savefig('result.svg')
plt.show()
