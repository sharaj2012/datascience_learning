#%%
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import metrics

# %%
#Data Preprocessing
#Preprocessing the training set -> to avoide overfeeding
train_datagen = ImageDataGenerator(
        rescale=1./255, #rescaling(normalization) since pixel values range from 0 to 255, this will bring all values between 0 & 1 
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
train_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

# %%
#Preprocessing the Test Set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

# %%
#Building the CNN
#initialising the CNN
cnn = tf.keras.models.Sequential()

#Step1: Convolution
#cnn.add(tf.keras.layers.Conv2D(filters=32,keral_size=3,activation='relu',input_shape=[64,64,3]))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))


#Step2: Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#Addiing the second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#Step3: Flattening
cnn.add(tf.keras.layers.Flatten())

#Step4: Full Connecting
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))

#Step5: Output Layer
cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

# %%
#Training the CNN
#Compiling the CNN
cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Training the CNN on the Training set and evaluating it on the Test Set
cnn.fit(x=train_set,validation_data=test_set,epochs=25)

#%%
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = cnn.predict(test_image)
train_set.class_indices
if result[0][0] == 1:
   prediction ='dog'
else:
   prediction ='cat'

print(prediction)

# %%
