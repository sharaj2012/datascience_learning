#%%
#frequent used imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
#importing a csv file
df_insurance=pd.read_csv('data/insurance.csv')
df_titanic=pd.read_csv('data/titanic.csv')
#getting the shape of the table
print(df_insurance.shape)
#getting the name of the columns
print(df_insurance.columns)

# %%
#checking what column has how many null values
df_titanic.isnull().sum()

# %%
#%matplotlib inline

# %%
x=df_titanic['Age'].dropna()  #drops NaN value and displays the age column only with no NaN values
x=df_titanic[['Age','Cabin']].dropna()    #drops NaN value fro both Age and Crew and displays the column only with no NaN values
x
# %%
df_titanic[df_titanic['Age'].isnull()]    #displays the entire dataframe only with values where Age is Nan
# %%
df_titanic[df_titanic['Age'].isnull()].index    #gives the index of the null values

# %%
import requests
import pandas as pd
url = 'https://raw.githubusercontent.com/FBosler/Medium-Data-Extraction/master/sales_team.csv'
res = requests.get(url, allow_redirects=True)
with open('sales_team.csv','wb') as file:
    file.write(res.content)
sales_team = pd.read_csv('sales_team.csv')

# %%
# Why set a value for “random state”?
# Ensures that a random process will output the same results every time, which makes your code reproducible (by you or by others)

from sklearn.model_selection import train_test_split 

# any positive integer can be used for the random_state value
X_train, X_test,y_train,y_test =  train_test_split(X,y,test_size=0.5,random_state=1)

#using the same random_state value results in the same random split

# %%
