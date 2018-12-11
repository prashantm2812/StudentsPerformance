# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 22:09:03 2018

@author: Prashant Maheshwari
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('StudentsPerformance.csv')
X = dataset[['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']]
Y = dataset[['math score', 'reading score', 'writing score']]

X = pd.get_dummies(X, columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'])

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.10, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
sc1 = StandardScaler()
Y_train = sc1.fit_transform(Y_train)
Y_test = sc1.transform(Y_test)


import keras
from keras.models import Sequential
from keras.layers import Dense

def create_model():
    model = Sequential()
    model.add(Dense(units = 17, kernel_initializer = 'uniform', activation = 'relu', input_dim = 17))
    model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'linear'))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

seed = 7
np.random.seed(seed)

model = KerasClassifier(build_fn = create_model, epochs = 100, batch_size = 5, verbose = 50)

from sklearn.model_selection import GridSearchCV as gscv
batch_size = [32, 64 ,100]
epochs = [25, 50, 100, 200, 150]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = gscv(estimator=model, param_grid=param_grid,  verbose = 60, n_jobs= -1)

grid_search = grid.fit(X_train, Y_train)

grid_search.best_score_#0.7499999933772616
grid_search.best_params_#{'batch_size': 100, 'epochs': 200}
