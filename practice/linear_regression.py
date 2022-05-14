
#? We will build a Linear regression model for Medical cost dataset. The dataset consists of age, sex, BMI(body mass index), children, smoker and region feature, which are independent features and charge as a dependent feature. We will predict individual medical costs billed by health insurance.

#! Definition and Working Principle
#? Linear regression is a supervised learning algorithm, its a very straightforward approach for predictio=ng a quantitative response Y on the basis of one or more ppredictors/independent variables.it work on the principle of Mean square errror(𝑀𝑆𝐸) . it's goal is to minimize sum of square difference between observed dependent variable in the given data set and those predicted by linear regression fuction.

#%%
#todo Import Library and Dataset
import pandas as pd #data manipulation
import numpy as np #data manipulation
import matplotlib.pyplot as plt #visualization
import seaborn as sns
from seaborn import palettes #visualization

# rcParams -> Each time Matplotlib loads, it defines a runtime configuration (rc) containing the      default styles,for every plot element you create. This configuration can be adjusted at any time using the plt. 
# figure -> https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.figure.html
plt.rcParams['figure.figsize']=[8,5] 
# figsize(float,float):width,height in inches.if not provided,  
# defaults to rcParams["figure.figsize"] = [6.4, 4.8].

#%%
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight']= 'bold'
plt.style.use('seaborn-whitegrid')

#%%
#todo import dataset
df = pd.read_csv('data/insurance.csv')
print('\nNumber of rows and columns in the data set',df.shape)
print('')

#todo looking into top columns and rows
df.head()

# now we have a dataset. The shape is (1338,7). That means m=1338 training examples and n=7 independen variables
# we have Age,Sex,BMI,Children,Smoke,Region as independent variable and Charges as Dependent variable

#%%
#todo for our visualization purpose will fit line using seaborn library only for bmi as independent variable and charges as dependent variable
sns.lmplot(x='bmi',y='charges',data=df,aspect=2,height=6,line_kws={'color':'red'}) 
# aspect is width by heigth ratio
# lmplot: linear model plot

plt.xlabel('Boby Mass Index$(kg/m^2)$: as Independent variable')
plt.ylabel('Insurance Charges: as Dependent variable')
plt.title('charge Vs BMI')

# On top we have a scatter plot and regression line and the shaded portion is the confidence interval based on bootstrapping

#! Exploratory Data Analysis
#%%
#todo describe() gives the count,mean,standard deviation,min,max,25%,50%,75%, and max value of all the columns
df.describe() 
# example of age-> 25 percentile means -> 25% of age is less than 27
# 50 percentile means -> 50% of the data is less than 39
# 75 percentile means -> 75% of the data is less than 51

#%%
#todo plotting heatmap for checking if there is any null value in the columns
f = plt.figure(figsize=(12,4))
sns.heatmap(df.isnull(),cmap='viridis',cbar=False,yticklabels=False)
plt.title('missing values HEatmap')

#todo gives the sum of all the null values in the specific columns
df.isnull().sum()

#%%
#todo checing the correlation among all the independent variables
corr= df.corr()
sns.heatmap(corr,annot=True)
# it looks like there is no storng correlation among the variables

#%%
#todo distribution plot of charge
f = plt.figure(figsize=(15,5))

ax=f.add_subplot(121)
sns.distplot(df['charges'],bins=40,color='r')
ax.set_title('Distribution of insurance charges')
#looks like charges is left skewed

#we use log transformation to make it gaussian dstributed
ax=f.add_subplot(122)
sns.distplot(np.log10(df['charges']),bins=40,color='b',ax=ax)
ax.set_title('dist in LOG scale')
ax.set_xscale('log')
# it looks like in the left plot, the charges varies from 1120 and 63500, the plot is right skewed 
# in right plot, we applied natural log then plot tends to have normal distribution**

#%%
#todo violin plot for sex and smoker vs charges
f=plt.figure(figsize=(15,6))
ax=f.add_subplot(121)
sns.violinplot(x='sex',y='charges',data=df,palette='Wistia',ax=ax)
ax.set_title('violin plot of charges vs sex')
# what we can observe is both male and female has the same charges ranging from 0 to 52000
# but for female its more broader from 0 to 20000
  
ax=f.add_subplot(122)
sns.violinplot(x='smoker',y='charges',data=df,palette='magma',ax=ax)
ax.set_title('violin plot of charges vs smoking')
# what we see here is charges for non smokers rangers only from 0 to 22000
# but for smokers 

# In[8]:
plt.figure(figsize=(14,6))
sns.boxplot(x='children',y='charges',hue='sex',palette='rainbow',data=df)

# In[29]:
plt.figure(figsize=(14,6))
sns.boxplot(x='children',y='charges',palette='rainbow',data=df)

# In[37]:
df.groupby('children').agg(['mean','min','max'])['charges']

# In[ ]:
plt.figure(figsize=(14,6))
sns.violinplot(x='region',y='charges',hue='sex',data=df,palette='rainbow',split=True)
plt.title('Violin plot of charges vs children')

# %%
f = plt.figure(figsize=(20,8))
ax = f.add_subplot(121)
sns.scatterplot(x='age',y='charges',data=df,palette='magma',hue='smoker',ax=ax)
ax.set_title('Scatter plot of Charges vs age')

ax = f.add_subplot(122)
sns.scatterplot(x='bmi',y='charges',data=df,palette='viridis',hue='smoker')
ax.set_title('Scatter plot of Charges vs bmi')

# %%

df['children'].unique() 

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),['sex','children','smoker','region'])],remainder='passthrough')
df_encode = np.array(ct.fit_transform(df))

# %%
df_encode.shape
df.shape
# %%
type(df_encode)
# %%
df_encode
# %%
df
# %%
categorical_columns=['sex','children','smoker','region']
df_new1=pd.get_dummies(data=df,columns=categorical_columns,drop_first=True)
# %%
df_new1
# %%
df_new1.shape
# %%
df_new1.shape
# %%
