# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 12:44:33 2021

@author: Hunny
"""

#%% Importing libraries
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification as create
from sklearn.preprocessing import StandardScaler as ss
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix as cm, accuracy_score as acc, classification_report as cr
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#%%
# Creating Dataset
x,y = create(n_samples = 1000, n_classes = 2)
#%%
# Visualizing Dataset
plt.figure(figsize=(7,6))
plt.scatter(x[:,0],x[:,1],c=y,s=20,edgecolor='green')
plt.show()
#%%
# Scaling Dataset
x = ss().fit_transform(x)

# Splitting into training and testing data
x_train, x_test, y_train, y_test = tts(x,y,test_size=0.2)

# Loading the regression model
model = lr()
# Training the model
model.fit(x_train,y_train)

# Predicting
y_pred = model.predict(x_test)
#%%
# Evaluating
print("Confusion matrix: \n",cm(y_test,y_pred))
print("Accuracy: {:.2%}".format(acc(y_test,y_pred)))
print("Classification Report: \n",cr(y_test,y_pred))
