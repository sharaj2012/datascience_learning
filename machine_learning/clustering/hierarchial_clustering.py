#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

# %%
#Using the dendrogram t find the optimal number of clusters
from scipy.cluster import hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

# %%
#Training the Hierarchal Clustering model on the datatset

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(X)

# %%
print(y_hc)

# %%
#visualizing the cluster

plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c='red',label='Cluster1')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c='blue',label='Cluster2')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c='green',label='Cluster3')
#plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,c='cyan',label='Cluster4')
#plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,c='magenta',label='Cluster5')
plt.legend()
plt.show()
# %%
