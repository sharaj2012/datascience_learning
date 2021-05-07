#%%
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

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
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
