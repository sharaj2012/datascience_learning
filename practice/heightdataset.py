#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # Visualization
import seaborn as sns #Visualization
# %%
heightdataset = pd.read_csv('data/agevsheight.csv')
# %%
heightdataset

# %%
f= plt.figure(figsize=(12,4))

ax=f.add_subplot(121)
sns.distplot(heightdataset[]ik,bins=40,color='r',ax=ax)
ax.set_title('Distribution of insurance charges')

# %%
plt.title('Charge Vs BMI');