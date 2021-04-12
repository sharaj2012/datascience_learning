#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

# %%
print(x)
# %%
print(y)

# %%
# reshape salary to 2D Matrix to display vertically jst like the level colun and 
y = y.reshape(len(y),1)

# %%
#fetaure scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()

x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

# %%
print(x)

# %%
print(y)

# %%
# Training the SVR model on the whole dataset
# here we using Gaussian Radial Basis Function(RBF) kernel and also recommended
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x,y)

# %%
# predicting a new result
regressor.predict()