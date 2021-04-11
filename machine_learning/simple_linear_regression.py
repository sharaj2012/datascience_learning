#%%
# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_random_state

# %%
# importing the dataset
salary_dataset = pd.read_csv('Salary_Data.csv')

# %%
salary_dataset

# %%
x = salary_dataset.iloc[:,:-1].values
y = salary_dataset.iloc[:,-1].values

# %%
y

# %%
#splitting the dataset into training set and test set 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2,random_state=0) 

# %%
print(x_train)
 
 # %%
 # Training the simple Linear Regression model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# %%
# predicting the Test set results
y_pred = regressor.predict(x_test)

# %%
# visualizing the training set results
plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience(Training Set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# %%
# visualizing the test set result
plt.scatter(x_test,y_test,color = 'red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience(Test Set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
 # %%
# we are having such a beautiful graph because the dataset values are in linear relationship