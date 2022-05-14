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
