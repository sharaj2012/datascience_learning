#%%
from operator import imod
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
mnist.keys()
# %%
mnist['data']
mnist['data'].shape
mnist['target'].shape
# %%
X,y = mnist['data'],mnist['target']
# X[0].shape

X_final = X[2].reshape(28,28)
X_final.shape
# X.shape
# y.shape
# %%
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.imshow(X_final,cmap="binary")
plt.axis("off")
plt.show()
# %%
plt.figure(figsize=(10,10)) # specifying the overall grid size

for i in range(100):
    plt.subplot(10,10,i+1)    # the number of images in the grid is 5*5 (25)
    X_final = X[i].reshape(28,28)
    plt.imshow(X_final,cmap="binary")
    plt.axis("off")

plt.show()
# %%
type(y[2])

# %%
import numpy as np
y_int = y.astype(np.uint8)
y_int
# %%
X_train,X_test,y_train,y_test = X[:60000],X[60000:],y_int[:60000],y_int[60000:]
# %%
X_train.shape
y_train.shape
# %%
y_train_4 = (y_train==4)
y_test_4 = (y_test==4)

# %%
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train,y_train_4)
# %%
# %%
import matplotlib.image as img
image = img.imread("four.png")
image.shape
image_2d = image.reshape(784)
image_2d.shape

# %%
for x in range(10):
    print(x)
    sgd_clf.predict([X[i]])
# %%
sgd_clf.predict([X[2]])
# %%
#Performance Measures
#cross validation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from statistics import mean
accuracy = []
skfld = StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
accuracy = []
for train_index,test_index in skfld.split(X_train,y_train_4):
    X_train_folds,X_test_folds = X_train[train_index],X_train[test_index]
    y_train_folds,y_test_folds = y_train_4[train_index],y_train_4[test_index]
    sgd_clf.fit(X_train_folds,y_train_folds)
    y_pred = sgd_clf.predict(X_test_folds)
    score = accuracy_score(y_test_folds,y_pred)
    accuracy.append(score)
    print(score)
accuracy
mean(accuracy)

# %%
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf,X_train,y_train_4,cv=10,scoring="accuracy")

# %%
X.shape
# %%
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf,X_train,y_train_4,cv=10)
y_train_pred
# %%
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_4,y_train_pred)
# %%
from sklearn.metrics import precision_score,recall_score
print('precision',precision_score(y_train_4,y_train_pred))
print('recall',recall_score(y_train_4,y_train_pred))
# %%
from sklearn.metrics import precision_score,recall_score

recall_score(y_train_4,y_train_pred)
# %%
