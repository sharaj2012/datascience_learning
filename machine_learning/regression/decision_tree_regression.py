#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import RegressorMixin

# %%
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

# %%
# training the decision tree Regression model on the whole dataset

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

# %%
# predict a new result

regressor.predict([[6.5]])

# %%
#visualising the decisison tree regression in high resolution
# its a step graph since whta decision tree does is takes the average and plots
x_grid = np.arange(min(x),max(x),0.1 )
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color = 'blue')
plt.title('Decision Tress Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
# %%
