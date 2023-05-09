# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 17:40:10 2022

@author: MLBA
"""



import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import keras
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization, Input
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau
import cv2
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.utils import shuffle

import ipywidgets as widgets
import io
from PIL import Image
import tqdm
from IPython.display import display,clear_output


def visualize(direction):
    list_dir=os.listdir(direction)
    plt.figure(figsize=(14,8))
    for i in range(1,7):
        plt.subplot(2,3,i)
        img= plt.imread(os.path.join(direction,list_dir[i]))
        plt.imshow(img,cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    
#데이터의 시각화
    
MildDemented_dir= 'C:/Users/jimflower/Desktop/Alzheimer_s Dataset/test/MIldDemented'
visualize(MildDemented_dir)

ModerateDemented_dir= 'C:/Users/jimflower/Desktop/Alzheimer_s Dataset/test/ModerateDemented'
visualize(ModerateDemented_dir)

NonDemented_dir= 'C:/Users/jimflower/Desktop/Alzheimer_s Dataset/test/NonDemented'
visualize(NonDemented_dir)

VeryMildDemented_dir= 'C:/Users/jimflower/Desktop/Alzheimer_s Dataset/test/VeryMildDemented'
visualize(VeryMildDemented_dir)

BASE_DIR = 'C:/Users/jimflower/Desktop/Alzheimer_s Dataset'
TRAIN_DIR = 'C:/Users/jimflower/Desktop/Alzheimer_s Dataset/train'
TEST_DIR = 'C:/Users/jimflower/Desktop/Alzheimer_s Dataset/test'

CLASSES = [ 'NonDemented',
            'VeryMildDemented',
            'MildDemented',
            'ModerateDemented']

#경로 설정 및 클래스 설정

IMG_SIZE = 150
IMAGE_SIZE = [150, 150]
DIM = (IMG_SIZE, IMG_SIZE)

#이미지의 사이즈 설정

train =tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,validation_split=.2,subset='training',
    seed=1337,image_size=IMAGE_SIZE,batch_size=64)

validatioin =tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,validation_split=.2,subset='validation',
    seed=1337,image_size=IMAGE_SIZE,batch_size=64)

#데이터 트레인 셋과 밸리데이션 셋 구분 

class_names = ['MildDementia', 'ModerateDementia', 'NonDementia', 'VeryMildDementia']
num_classes=len(class_names)
train.class_names=class_names
validatioin.class_names=class_names


#각각의 클래스 네임 지정

from tensorflow.data.experimental import AUTOTUNE 
def one_hot_label(image, label):
    label = tf.one_hot(label, num_classes)
    return image, label

#원핫 함수

train=train.map(one_hot_label, num_parallel_calls=AUTOTUNE)
validatioin=validatioin.map(one_hot_label, num_parallel_calls=AUTOTUNE)
#이미지에 원핫 을 사용하여 숫자로 처리하기 위함

train= train.cache().prefetch(buffer_size=AUTOTUNE)
validatioin = validatioin.cache().prefetch(buffer_size=AUTOTUNE)
#cache preprocessing 시간을 줄이는 코드이며 prefetch는 데이터의 로드 시간을 줄이기 위해 사용


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * (0.1 **(epoch / s))
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(0.01, 20)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
# 에폭별로 학습률을 조정하여 오버피팅 및 언더피팅 방지 및 최적의 모델에서 빠져나감 방지
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("alzheimer_model.h5",
                                                    save_best_only=True)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,
                                                     restore_best_weights=True)
#10회 이상 손실함수가 줄어 들지 않을때 중지 또한 restore~ 를 설정해줬기 때문에 최상의 가중치를 복원해주는 코드


#첫모델
# def build_model():
#     model = tf.keras.Sequential([
#         tf.keras.Input(shape=(150,150, 3)),
        
        
#         tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
#         tf.keras.layers.MaxPool2D(2,2),
        
#         tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
#         tf.keras.layers.MaxPool2D(2,2),

        
#         tf.keras.layers.Flatten(),
        
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dropout(0.3),
        
#         tf.keras.layers.Dense(4, activation='softmax')
#     ])
    
#     return model

#두번째모델
# def build_model():
#     model = tf.keras.Sequential([
#         tf.keras.Input(shape=(150,150, 3)),
        
        
#         tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
#         tf.keras.layers.MaxPool2D(2,2),
        
#         tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
#         tf.keras.layers.MaxPool2D(2,2),
        
#         tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
#         tf.keras.layers.MaxPool2D(2,2),
        
#         tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
#         tf.keras.layers.MaxPool2D(2,2),
        
#         tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
#         tf.keras.layers.MaxPool2D(2,2),
        
#         tf.keras.layers.Flatten(),
        
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dropout(0.3),
#         tf.keras.layers.Dense(4, activation='softmax')

#     ])
    
#     return model

#세번째~다섯번째모델
# def build_model():
#     model = tf.keras.Sequential([
#         tf.keras.Input(shape=(150,150, 3)),
        
        
#         tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.MaxPool2D(2,2),
        
#         tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.MaxPool2D(2,2),
        
#         tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.MaxPool2D(2,2),
        
#         tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.MaxPool2D(2,2),
        
#         tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.MaxPool2D(2,2),
        
#         tf.keras.layers.Flatten(),
        
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dropout(0.3),
#         tf.keras.layers.Dense(4, activation='softmax')


#     ])
    
#     return model


#여섯번째모델
# def build_model():
#     model = tf.keras.Sequential([
#         tf.keras.Input(shape=(150,150, 3)),
        
        
#         tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.MaxPool2D(2,2),
        
#         tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.MaxPool2D(2,2),
        
        
#         tf.keras.layers.Flatten(),
        

#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dropout(0.3),
#         tf.keras.layers.Dense(4, activation='softmax')


#     ])
    
#     return model








# 최종 모델
def build_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(150,150, 3)),
        
        
     
        tf.keras.layers.Conv2D(16, (3, 3),padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        
        tf.keras.layers.Conv2D(32, (3, 3),padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
    
        tf.keras.layers.Flatten(),
        

        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    
    return model
#우리의 모델


model = build_model()


model.summary()

#모델 요약

METRICS = [tf.keras.metrics.AUC(name='auc')]
#모델 평가 지표

model.compile(
        optimizer='adam',
        loss=tf.losses.CategoricalCrossentropy(),
        #metrics=METRICS #Auc지표
        metrics = ['accuracy']
        )
#모델 컴파일 

history = model.fit(
   train,validation_data=validatioin,callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler],epochs=50
)

#훈련 및 에폭수 설정

fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()


for i, met in enumerate(['accuracy', 'loss']):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history['val_' + met])
    ax[i].set_title('Model {}'.format(met))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(met)
    ax[i].legend(['train', 'val'])

#모델 정확도 에폭별 시각화

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "C:/Users/jimflower/Desktop/Alzheimer_s Dataset/test",
    image_size=IMAGE_SIZE,
)
#테스트 셋 불러오기

test_ds = test_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

_ = model.evaluate(test_ds)
#테스트 데이터 결과
model.save('Alzheimer_model.h5')
#모델 저장