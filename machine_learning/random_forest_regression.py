#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

# %%
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10,random_state=0)
# n_estimators = number of trees in the forest
regressor.fit(x,y)

# %%
regressor.predict([[6.5]])

# %%
x_grid = np.arange(min(x),max(x),0.1 )
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color = 'blue')
plt.title('Decision Tress Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
# %%
