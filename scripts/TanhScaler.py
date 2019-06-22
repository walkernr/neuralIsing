# -*- coding: utf-8 -*-
"""
Created on Thu Jun 06 01:11:08 2018

@author: Nicholas
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


# tanh function with range (beta0, )
tanhScaler = lambda x, beta: beta[0]*(np.tanh(x)+beta[1])
tanhScalerInv = lambda x, beta: np.arctanh(x/beta[0]-beta[1])

# tanh scaler class
class TanhScaler:
    ''' this scaler feeds the z-scre from the standard scaler into a tanh function

        the tanh function allows for the output to be less sensitive to outliers and maps
        all features to a common numerical domain '''

    def __init__(self, feature_range=(0.0, 1.0)):
        ''' initialize standard scaler '''
        self.standard = StandardScaler()
        u, v = feature_range
        a, b = 0.5*(v-u), (u+v)/(v-u)
        self.beta = a, b

    def fit(self, X):
        ''' fit standard scaler to data X '''
        self.standard.fit(X)

    def transform(self, X):
        ''' transform data X '''
        zscore = self.standard.transform(X)  # tranform with standard scaler first
        return tanhScaler(zscore, self.beta)  # return tanh scaled data

    def fit_transform(self, X):
        ''' simultaneously fit and transform data '''
        self.fit(X)  # fit first
        return self.transform(X) # return transform output

    def inverse_transform(self, TZ):
        ''' inverses fit '''
        zscore = tanhScalerInv(TZ, self.beta)
        return self.standard.inverse_transform(zscore)
