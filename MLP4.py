#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn import metrics
#pre processing

# using premierleaguelong.csv

filename = './DataFrameForNeuralWorkingCopy (1).csv'
dataset = pd.read_csv(filename)


print(dataset.shape)
dataset = dataset.values
features = dataset[:, 15:19]
labels = dataset[:,-1]
print(features)
print(features.shape,labels.shape)
X = features
y = np.int_(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=0)

clf = MLPClassifier(random_state=0, max_iter=200).fit(X_train, y_train)#
y_pred = clf.predict(X_test)
f1_score_micro = f1_score(y_test, y_pred, average='micro')
f1_score_macro = f1_score(y_test, y_pred, average='macro')
print('MLP score:', f1_score_micro, f1_score_macro)

print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
print()

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0).fit(X_train, y_train)
y_pred = clf.predict(X_test)
f1_score_micro = f1_score(y_test, y_pred, average='micro')
f1_score_macro = f1_score(y_test, y_pred, average='macro')
print('Gradient Boosting score:', f1_score_micro, f1_score_macro)

print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
print()

clf = RandomForestClassifier(max_depth=2, random_state=0).fit(X_train, y_train)
y_pred = clf.predict(X_test)
f1_score_micro = f1_score(y_test, y_pred, average='micro')
f1_score_macro = f1_score(y_test, y_pred, average='macro')
print('Random Forest score:',f1_score_micro, f1_score_macro)


print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
print()


# In[ ]:




