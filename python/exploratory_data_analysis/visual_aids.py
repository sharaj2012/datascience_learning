#%%
from datetime import datetime
import random
import pandas as pd
import radar

#%%
stockval = random.randint(100,500)
print(stockval)
date = radar.random_date(startdate,enddate)
print(date)
#%%
def generateData(n):
  startdate = datetime(2010,2,12)
  enddate = datetime(2010,4,12)
  stockdetail=[]
  for _ in range(n):
    stockval = random.randint(100,500)
    date = radar.random_date(startdate,enddate).strftime("%Y-%m-%d")
    stockdetail.append([date,stockval])
  
  df = pd.DataFrame(stockdetail,columns=['Date','Price'])
  df['Date'] = pd.to_datetime(df['Date'],format="%Y-%m-%d")
  df = df.groupby(by='Date').mean()
  return df

# %%
df= generateData(50)
df.head(10)

# %%
df

# %%
from pylab import rcParams
import matplotlib.pyplot as plt
# %%

rcParams['figure.figsize']=[14,10]
plt.plot(df)
plt.show()
# %%
plt.rcParams