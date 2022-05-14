#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
dataset =  pd.read_csv('data/Data.csv')

# %%
dataset

# %%
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values 

#%%
print(x)

# %%   
# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
#imputer.fit(x[:,1:3]) # to connect the object to the matrix of features x 
x[:,1:3] = imputer.fit_transform(x[:,1:3]) # apply the transformation

#%%
print(x)

# %%
# Encoding the independent variable
# using One Hot Encoding we turn county column into 3 columns, in this example France will have code 100,Spain = 010, Germany= 001, so there is no numerical order between thises countries, so ML model wont find any relation between these crea

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
x = np.array(ct.fit_transform(x)) 

# Setting remainder='passthrough' will mean that all columns not specified in the list of “transformers” will be passed through without transformation, instead of being dropped. Once the transformer is defined, it can be used to transform a dataset

# %%
print(x)

# %%
## Encoding the Dependent Variable

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# %%
print(y)

# %%
lex = LabelEncoder()
lex.fit(["amsterdam", "paris","tokyo", "tokyo"])
lex.transform(["amsterdam", "tokyo", "tokyo"])

# %%
# splitting dataset into Training and Test Set
# Feature Scaling - Do we need to apply feature scaling before or after splitting the dataset--------Ans: Feature scaling sud be done after splitting the dataset, because we sud not apply feature scaling on the test set since its suppose to be something of a new observation
# Training set -> where ML model is going to train
# Test set -> where you gonna evaluate the performance of the ML model on new observations
# Feature Scaling means scaling all the variables/features to make sure they all take values in the same scale, and we do this to prevent one feature dominate the other
# we basically going to get four sets, 
# x_train = matrix of feattures of training set
# y_train = dependent variable of the training set
# x_test & y_test respectively
# x_train & y_train are expected in a method called the fit method and for the predictions also called as Inference

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y ,test_size = 0.2, random_state = 1)
# x,y = independent,dependent variable
# test_size = how much % of dataset as Test Set
# random_state = 1, so that the test_set and traing_Set gets randomly created

print(x_train)
print(x_test)
print(y_train)
print(y_test)

# %%
# how are we going to apply feature scaling tenchiques -> Standardisation | Normalisation
# standardisation work all the time
# Q. do we have to apply feature scaling to the dummy variables in the matrix of features i.e "1.0 0.0 0.0" in [1.0 0.0 0.0 44.0 72000.0].....Ans. No since they already between 0 & 1, and if we apply standardisation,we will get some rubbish value which wont help

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:,3:] = sc.fit_transform(x_train[:,3:])
x_test[:,3:] = sc.transform(x_test[:,3:]) # here we using standardisation we did on x_train and applied it on x_test,we didnot create a new sdandardisation

# %%
print(x_train)

# %%
print(x_test)
# %%
 