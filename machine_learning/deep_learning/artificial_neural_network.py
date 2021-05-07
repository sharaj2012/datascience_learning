#%%
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import metrics

# %%
tf.__version__

# %%
# Data Preprocessing
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values

print(X)
print(y)

# %%
#Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])

# %%
print(X)

# %%
#OneHotEncoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X = np.array(ct.fit_transform(X)) 

# %%
print(X)

# %%
print(X[1,:])

# %%
# Splitting the dataset into Training and Test Set & then appliying Feature Scaling to every variable of the dataset SINCE ITS MUST IN DEEP LEARNING
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y ,test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# %%
#Building the ANN
ann = tf.keras.models.Sequential()

#Adding the input layer and first hidden layer
#how many unit variables(the hidden layer neurons) do we need to have? ans-> there is no rule of thumb, its just based on experimentation with different hyper paramenters... will give units as something relevent
# relu -> rectifier activation function
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

#Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

#Adding the output layer
# since th eoutput is binary, only one output neuron is suffecient 
# sigmoid activation function -> when doin binary classification 
# predicting more than two categories -> softmax classification
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

#Training the ANN
#compiling the ANN
#adam -> this optimizer can perform Stochastic Gradient Descent
#for binary classification-> binary_crossentropy loss
#for non binary classification-> categorical_crossentropy loss
ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Training the ANN on the Training set
ann.fit(X_train,y_train,batch_size=32,epochs=100)

# %%
# Predicting the result of a single observation
print(ann.predict(sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]])))

# %%
print(ann.predict(sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]]))>0.5)

# %%
#Predicting the TEst set results
y_pred = ann.predict(X_test)
y_pred = (y_pred >0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# %%
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
# %%
  