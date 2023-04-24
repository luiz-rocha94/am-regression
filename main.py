# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 13:55:45 2023

@author: rocha
"""

import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
from sklearn.metrics import r2_score

# load data.
file = '../cw1/boston.csv'
data = pd.read_csv(file)

# split data.
X = data.values[:,:-1]
y = data.values[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, 
                                                    random_state=0)

# cross-validation.
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

# model.
model = linear_model.LinearRegression()
parameters = {'positive':[True, False]}

# grid search.
scoring = ['neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']
clf = GridSearchCV(model, parameters, scoring=scoring, cv=cv, refit='r2')
clf.fit(X_train, y_train)
best_paramns = clf.best_params_

# final model
model = linear_model.LinearRegression(**best_paramns)
model.fit(X_train, y_train)
y_test_hat = model.predict(X_test)
score = r2_score(y_test, y_test_hat)


