#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# %%
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
x = np.array(ct.fit_transform(x)) 

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# %%
# for multiple linear regression,  there is no need to apply feature scaling , bcoz there are coeffecients multiplied to each independent variable of each feature and therefore it doesnt matter if some feature have high values than another because the coeffecients will compensate to put evenrything on the smae scale

# %%
# Training the Multiple Linear REgressing Model on the Training set
# Do we need to do something to avoide Dummy Varibale TRap -> NOO , MultipleRegression class will only avoide the dummy variables so no need to worry
# do we need to apply Backward Elimination -> NOO, the sklearn class itself will automatically identify the best features that have the highest P value or are most statistically significant

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# %%
# Predicting the Test set results
# 1st one being the vector of the real profit and test set and 2nd one being the vector of the predicted profit and the test set

y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2) 
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

# %%
