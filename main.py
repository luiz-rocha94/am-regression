# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 13:55:45 2023

@author: rocha
"""

import pandas as pd
import numpy as np
from sklearn import covariance, linear_model, svm
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

import seaborn as sns
import matplotlib.pyplot as plt


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

# models.
models = {'linear':{'func':linear_model.LinearRegression, 
                    'params':{}}, 
          'ridge':{'func':linear_model.Ridge, 
                   'params':{'alpha':np.linspace(0.1,1,10,dtype=np.float32)}}, 
          'lasso':{'func':linear_model.Lasso, 
                   'params':{'alpha':np.linspace(0.1,1,10,dtype=np.float32)}}, 
          'elasticnet':{'func':linear_model.ElasticNet, 
                        'params':{'alpha':np.linspace(0.2,1,5,dtype=np.float32),'l1_ratio':np.linspace(0.05,0.1,2,dtype=np.float32)}}, 
          'svm':{'func':svm.SVR, 
                 'params':{'C':np.linspace(0.2,1,5,dtype=np.float32),'epsilon':np.linspace(0.05,0.1,2,dtype=np.float32)}}}
scores = {'models':list(models.keys()),
          'params':[], 'r2':[], 'mse':[], 'rmse':[]}

for key in models.keys():
    # model.
    model = models[key]['func']
    parameters = models[key]['params']

    # grid search.
    scoring = ['neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']
    clf = GridSearchCV(model(), parameters, scoring=scoring, cv=cv, refit='r2')
    clf.fit(X_train, y_train)
    results = pd.DataFrame(clf.cv_results_)[['params', 
                                             'mean_test_neg_mean_squared_error',
                                             'mean_test_neg_root_mean_squared_error',
                                             'mean_test_r2']]
    results['params'] = [f'{row}' for row in results['params']]
    best_paramns = clf.best_params_
    
    # plot results
    fig,ax = plt.subplots()
    sns.barplot(x='mean_test_r2', y='params', data=results, ax=ax)
    ax.bar_label(ax.containers[0], fmt='%.5f')
    mu, std = results['mean_test_r2'].mean(), np.nan_to_num(results['mean_test_r2'].std())
    ax.set(xlim=(mu-3*std, mu+3*std))
    ax.set(title=f'{key} mean 5-fold')
    plt.show()
    
    # final model
    model = model(**best_paramns)
    model.fit(X_train, y_train)
    y_test_hat = model.predict(X_test)
    r2 = r2_score(y_test, y_test_hat)
    mse = mean_squared_error(y_test, y_test_hat)
    rmse = mean_squared_error(y_test, y_test_hat, squared=False)
    scores['r2'].append(r2)
    scores['mse'].append(mse)
    scores['rmse'].append(rmse)
    scores['params'].append(f'{key} {best_paramns}'.replace('{','(').replace('}',')').replace(':','='))

scores_data = pd.DataFrame(scores)
for metric in ['r2', 'mse', 'rmse']:
    fig,ax = plt.subplots()
    sns.barplot(x=metric, y='params', data=scores_data, ax=ax)
    ax.bar_label(ax.containers[0], fmt='%.5f')
    mu, std = scores_data[metric].mean(), scores_data[metric].std()
    ax.set(xlim=(0, mu+3*std))
    ax.set(title='Models best params')
    plt.show()
