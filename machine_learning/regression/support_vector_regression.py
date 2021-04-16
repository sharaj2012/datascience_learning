#%%
from matplotlib import colors
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
print(y.ravel())

# %%
# Training the SVR model on the whole dataset
# here we using Gaussian Radial Basis Function(RBF) kernel and also recommended
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x,y.ravel())

# %%
# predicting a new result
sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]]))) 
# [[]] bcoz its 2d matrix and inver transform bcoz we need not the scaling but actiual value

# %%
# visualizing the SVR reslts
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y),color='red')
plt.plot(sc_x.inverse_transform(x),sc_y.inverse_transform(regressor.predict(x)),color = 'blue')
plt.title('SVR Model')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

# %%
# visualising the SVR results(for higher resolution and smoother curve1) 
x_grid = np.arange(min(sc_x.inverse_transform(x)),max(sc_x.inverse_transform(x)),0.1 )
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y),color='red')
plt.plot(x_grid,sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))),color = 'blue')
plt.title('Support Vector Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
# %%
