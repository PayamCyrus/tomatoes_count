# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 12:19:54 2022

@author: Payam_(cyrus)
"""
#load my dataset and csvs
import pandas as pd
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import PIL
from PIL import Image
from keras.models import Sequential

from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np

df = pd.read_csv(
   r"C:\Users\Payam_(cyrus)\Desktop\porojec_machin_vision\poroje\payam.csv", 
    na_values=['NA', '?'])

df['tomato_img_name']=df["tomato_img_name"].astype(str)

#Separate into a training and validation 

TRAIN_PCT = 0.9
TRAIN_CUT = int(len(df) * TRAIN_PCT)

df_train = df[0:TRAIN_CUT]
df_validate = df[TRAIN_CUT:]

print(f"Training size: {len(df_train)}")
print(f"Validate size: {len(df_validate)}")


df_train
df_validate
#This is what links the .csv with the images.

IMAGES_DIR =r"C:\Users\Payam_(cyrus)\Desktop\porojec_machin_vision\poroje\payam dataset tomatoes24000"

#------------------------------------------------------------------------------
# import matplotlib.pyplot as plt
# import os 
# sub_class = os.listdir(IMAGES_DIR)

# fig = plt.figure(figsize=(10,5))
# for e in range(len(sub_class[:8])):
#     plt.subplot(2,4,e+1)
#     img = plt.imread(os.path.join(IMAGES_DIR,sub_class[e]))
#     plt.imshow(img, cmap=plt.get_cmap('gray'))
#------------------------------------------------------------------------------

training_datagen = ImageDataGenerator(
  rescale = 1./255,
  horizontal_flip=True,
  vertical_flip=True,
  fill_mode='nearest')

train_generator = training_datagen.flow_from_dataframe(
        dataframe=df_train,
        directory=IMAGES_DIR,
        x_col="tomato_img_name",
        y_col="toomato count",
        target_size=(128,128),
        batch_size=2,
        class_mode='other')

validation_datagen = ImageDataGenerator(rescale = 1./255)

val_generator = validation_datagen.flow_from_dataframe(
        dataframe=df_validate,
        directory=IMAGES_DIR,
        x_col="tomato_img_name",
        y_col="toomato count",
        target_size=(128,128),
        class_mode='other')

#Deep learning lsyer's network and design 

import tensorflow as tf
from keras.models import Model 
from keras.callbacks import EarlyStopping
#from keras import activations
from keras.layers import  MaxPooling2D, BatchNormalization, Flatten, activation, Dense, Conv2D
from keras import layers 
import matplotlib as mpl
import numpy as np

input_layer = layers.Input(shape=(128,128,3))


L1= Conv2D(64, 7,padding='same', use_bias=True, kernel_initializer='glorot_uniform')(input_layer)
L2=BatchNormalization()(L1)
L3=layers.Activation('relu')(L2)

L4=MaxPooling2D((3,3),strides=2, padding='same')(L3)

L5=Conv2D(32, (1,1),padding='same',use_bias=True, kernel_initializer='glorot_uniform')(L4)
L6=BatchNormalization()(L5)
L7=layers.Activation('relu')(L6)

L7=Conv2D(192, (5,5),padding='same' ,use_bias=True, kernel_initializer='glorot_uniform')(L6)
L8=BatchNormalization()(L7)
L9=layers.Activation('relu')(L8)

L10=MaxPooling2D((3, 3),strides=2, padding='same')(L9)

#4. Modified Inception-ResNet-A module. 

L11=Conv2D(32, (1,1),padding='same', use_bias=True,kernel_initializer='glorot_uniform')(L10)       
L12=BatchNormalization()(L11)
L13=layers.Activation('relu')(L12)

L14=Conv2D(32, (1,1),padding='same', use_bias=True,kernel_initializer='glorot_uniform')(L10)
L15=BatchNormalization()(L14)
L16=layers.Activation('relu')(L15)
L17=Conv2D(32, (3,3),padding='same', use_bias=True,kernel_initializer='glorot_uniform')(L16)
L18=BatchNormalization()(L17)
L19=layers.Activation('relu')(L18)

L20=Conv2D(32, (1,1),padding='same', use_bias=True,kernel_initializer='glorot_uniform')(L10)
L21=BatchNormalization()(L20)
L22=layers.Activation('relu')(L21)
L23=Conv2D(32, (3,3),padding='same', use_bias=True,kernel_initializer='glorot_uniform')(L22)
L24=BatchNormalization()(L23)
L25=layers.Activation('relu')(L24)
L26=Conv2D(32, 3,padding='same', use_bias=True,kernel_initializer='glorot_uniform')(L25)
L27=BatchNormalization()(L26)
L28=layers.Activation('relu')(L27)

L29=tf.keras.layers.concatenate([L13, L19, L28],axis=1)
L30=Conv2D(192, (65,1), padding='valid', use_bias=True,kernel_initializer='glorot_uniform')(L29)
L31=BatchNormalization()(L30)
L32=layers.Activation('linear')(L31)
L33=layers.Add()([L32,L10])
L34=layers.Activation('relu')(L33)

#repet Modified Inception-ResNet-A module
L35=Conv2D(32, (1,1),padding='same', use_bias=True,kernel_initializer='glorot_uniform')(L34)       
L36=BatchNormalization()(L35)
L37=layers.Activation('relu')(L36)

L38=Conv2D(32, (1,1),padding='same', use_bias=True,kernel_initializer='glorot_uniform')(L34)
L39=BatchNormalization()(L38)
L40=layers.Activation('relu')(L39)
L41=Conv2D(32, (3,3),padding='same', use_bias=True,kernel_initializer='glorot_uniform')(L40)
L42=BatchNormalization()(L41)
L43=layers.Activation('relu')(L42)

L44=Conv2D(32, (1,1),padding='same', use_bias=True,kernel_initializer='glorot_uniform')(L34)
L45=BatchNormalization()(L44)
L46=layers.Activation('relu')(L45)
L47=Conv2D(32, (3,3),padding='same', use_bias=True,kernel_initializer='glorot_uniform')(L46)
L48=BatchNormalization()(L47)
L49=layers.Activation('relu')(L48)
L50=Conv2D(32, 3,padding='same', use_bias=True,kernel_initializer='glorot_uniform')(L49)
L51=BatchNormalization()(L50)
L52=layers.Activation('relu')(L51)

L53=tf.keras.layers.concatenate([L37, L43, L52],axis=1)
L54 = Conv2D(192, (65,1) ,padding='valid', use_bias=True,kernel_initializer='glorot_uniform')(L53)
L55 = BatchNormalization()(L54)
L56=layers.Activation('linear')(L55)
L57=layers.Add()([L34,L56])
L58=layers.Activation('relu')(L57)


#Modified reduction module
L52=MaxPooling2D((3,3),strides=(2))(L54)
L_edit=Conv2D(128,1,use_bias=True,kernel_initializer='glorot_uniform')(L52)

L53=Conv2D(192, 1, padding='valid', use_bias=True,kernel_initializer='glorot_uniform')(L58)
L54=BatchNormalization()(L53)
L55=layers.Activation('relu')(L54)
L56=Conv2D(128,3, strides=2, padding='valid', use_bias=True,kernel_initializer='glorot_uniform')(L55)
L57=BatchNormalization()(L56)
L58=layers.Activation('relu')(L57)

L59=Conv2D(192, 1, padding='same', use_bias=True,kernel_initializer='glorot_uniform')(L58)
L60=BatchNormalization()(L59)
L61=layers.Activation('relu')(L60)
L62=Conv2D(128,(1,7), padding='same', use_bias=True,kernel_initializer='glorot_uniform')(L61)
L63=BatchNormalization()(L62)
L64=layers.Activation('relu')(L63)
L65=Conv2D(128,(7,1), padding='same', use_bias=True,kernel_initializer='glorot_uniform')(L64)
L66=BatchNormalization()(L65)
L67=layers.Activation('relu')(L66)
L68=Conv2D(128,(3,3), padding='same', use_bias=True,kernel_initializer='glorot_uniform')(L67)
L69=BatchNormalization()(L68)
L70=layers.Activation('relu')(L69)

L71=tf.keras.layers.concatenate([L_edit, L58, L70],axis=1)


#repet Modified Inception-ResNet-A module
L72=Conv2D(32, (1,1),padding='same', use_bias=True,kernel_initializer='glorot_uniform')(L71)       
L73=BatchNormalization()(L72)
L74=layers.Activation('relu')(L73)

L75=Conv2D(32, (1,1),padding='same', use_bias=True,kernel_initializer='glorot_uniform')(L71)
L76=BatchNormalization()(L75)
L77=layers.Activation('relu')(L76)
L78=Conv2D(32, (3,3),padding='same', use_bias=True,kernel_initializer='glorot_uniform')(L77)
L79=BatchNormalization()(L78)
L80=layers.Activation('relu')(L79)

L81=Conv2D(32, (1,1),padding='same', use_bias=True,kernel_initializer='glorot_uniform')(L71)
L82=BatchNormalization()(L81)
L83=layers.Activation('relu')(L82)
L84=Conv2D(32, (3,3),padding='same', use_bias=True,kernel_initializer='glorot_uniform')(L83)
L85=BatchNormalization()(L84)
L86=layers.Activation('relu')(L85)
L87=Conv2D(32, 3,padding='same', use_bias=True,kernel_initializer='glorot_uniform')(L86)
L88=BatchNormalization()(L87)
L89=layers.Activation('relu')(L88)

L90=tf.keras.layers.concatenate([L74, L80, L89],axis=1)
L91=Conv2D(128, (91,1), padding='valid', use_bias=True,kernel_initializer='glorot_uniform')(L90)
L92=BatchNormalization()(L91)
L93=layers.Activation('linear')(L92)
L94=layers.Add()([L71,L93])
L95=layers.Activation('relu')(L94)

#repet Modified Inception-ResNet-A module
L96=Conv2D(32, (1,1),padding='same', use_bias=True,kernel_initializer='glorot_uniform')(L95)       
L97=BatchNormalization()(L96)
L98=layers.Activation('relu')(L97)

L99=Conv2D(32, (1,1),padding='same', use_bias=True,kernel_initializer='glorot_uniform')(L95)
L100=BatchNormalization()(L99)
L101=layers.Activation('relu')(L100)
L102=Conv2D(32, (3,3),padding='same', use_bias=True,kernel_initializer='glorot_uniform')(L101)
L103=BatchNormalization()(L102)
L104=layers.Activation('relu')(L103)

L105=Conv2D(32, (1,1),padding='same', use_bias=True,kernel_initializer='glorot_uniform')(L95)
L106=BatchNormalization()(L105)
L107=layers.Activation('relu')(L106)
L108=Conv2D(32, (3,3),padding='same', use_bias=True,kernel_initializer='glorot_uniform')(L107)
L109=BatchNormalization()(L108)
L110=layers.Activation('relu')(L109)
L111=Conv2D(32, 3,padding='same', use_bias=True,kernel_initializer='glorot_uniform')(L110)
L112=BatchNormalization()(L111)
L113=layers.Activation('relu')(L112)

L114=tf.keras.layers.concatenate([L113, L104, L98],axis=1)
L115=Conv2D(128, (91,1), padding='valid', use_bias=True,kernel_initializer='glorot_uniform')(L90)
L116=BatchNormalization()(L91)
L117=layers.Activation('linear')(L116)
L118=layers.Add()([L95,L117])
L119=layers.Activation('relu')(L94)

# ENDed layers---------------------
L120=layers.AveragePooling2D((3,3),strides=(3))(L119)
L121=layers.Dense(768,activation='relu')(L120)
L122=layers.Dropout(0.65)(L121)
flat = layers.Flatten()(L122)
out_Layer_p=layers.Dense(1, activation='linear')(flat)


#summery of model
my_model= Model(input_layer,out_Layer_p)
my_model.summary()

#Run model&network
epoch_steps = 4000 # needed for 2.2
validation_steps = len(df_validate)
my_model.compile(optimizer='adam', loss = 'mean_squared_error', metrics=['accuracy'])
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto',
        restore_best_weights=True)

trainig_run = my_model.fit(train_generator, verbose = 1,
                           validation_data=val_generator, callbacks=[monitor], epochs=3)


history = trainig_run.history

import matplotlib.pyplot as plt
accuracies = history['accuracy']
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.plot(accuracies)


#real test of network

df_test = pd.read_csv(
    r"C:\Users\Payam_(cyrus)\Desktop\porojec_machin_vision\poroje\data small tomato/so small test 5 image.csv", 
    na_values=['NA', '?'])

df_test['img_code']=df_test["img_code"].astype(str)

test_datagen = ImageDataGenerator(rescale = 1./255)

test_generator = validation_datagen.flow_from_dataframe(
        dataframe=df_test,
        directory=IMAGES_DIR,
        x_col="img_code",
        batch_size=1,
        shuffle=False,
        target_size=(128, 128),
        class_mode=None)

# Found 5000 validated image filenames.

test_generator.reset()
pred = my_model.predict(test_generator,steps=len(df_test))



df_submit = pd.DataFrame({'img_code':df_test['img_code'],'tomatoes_counted':pred.flatten()})

df_submit.to_csv(r"C:\Users\Payam_(cyrus)\Desktop\porojec_machin_vision\poroje\result of machine frouit counted/machine counting.csv",index=False)








