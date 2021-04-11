#%%
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

# %%
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

# %%
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

# %%
# Training the polynomial regression model on the whole dataset

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4) # y = b1 + b1X1 + b2X1^2 + b3X1^3 + b4X1^4
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

# %%
# visualising the linear regressing results

plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('position vs Salary')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

# %%
# visualising the Polynomial regression results

plt.scatter(x,y,color='red')
plt.plot(x,lin_reg2.predict(x_poly),color='blue')
plt.title('position vs Salary')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()
# %%
# predict a new result with Linear Regression

lin_reg.predict([[6.5]])

# %%
# predict a new result with polynomial Regression

lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
# %%
