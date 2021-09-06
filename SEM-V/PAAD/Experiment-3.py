# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 23:03:37 2021

@author: Hunny
"""

#%% Importing libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression as create
from sklearn.preprocessing import StandardScaler as ss
from sklearn.linear_model import LinearRegression as lr
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#%%
# Creating Dataset
x,y = create(n_samples = 1000, n_features = 1, noise = 10)
# Visualizing Dataset
plt.figure(figsize=(9,9))
plt.subplot(2,2,1)
plt.scatter(x,y)
plt.xlabel("Independent Variable")
plt.ylabel("Dependent Variable")
plt.subplot(2,2,2)
sns.distplot(x,label = "X Distribution")
plt.legend()
plt.subplot(2,2,3)
sns.distplot(y,label = "Y Distribution")
plt.legend()
plt.tight_layout()
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
print("Training accuracy: {:.2%}".format(model.score(x_train,y_train)))
print("Testing accuracy: {:.2%}".format(model.score(x_test,y_test)))
print("Mean Squared Error: {:.2}".format(mse(y_test,y_pred)))