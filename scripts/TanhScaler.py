# -*- coding: utf-8 -*-
"""
Created on Thu Jun 06 01:11:08 2018

@author: Nicholas
"""

from __future__ import division, print_function
import numpy as np
from sklearn.preprocessing import StandardScaler


# tanh function with range (a, b)
tanhScaler = lambda a, b, x: 0.5*((b-a)*np.tanh(x)+(a+b))
invTanhScaler = lambda a, b, x: np.arctanh((2*x-(a+b))/(b-a))

# tanh scaler class
class TanhScaler:
    ''' this scaler feeds the z-scre from the standard scaler into a tanh function

        the tanh function allows for the output to be less sensitive to outliers and maps
        all features to a common numerical domain '''

    def __init__(self, feature_range=(0, 1)):
        ''' initialize standard scaler '''
        self.a, self.b = feature_range
        self.standard = StandardScaler()

    def fit(self, X):
        ''' fit standard scaler to data X '''
        self.standard.fit(X)

    def transform(self, X):
        ''' transform data X '''
        zscore = self.standard.transform(X)  # tranform with standard scaler first
        return tanhScaler(self.a, self.b, zscore)  # return tanh scaled data

    def fit_transform(self, X):
        ''' simultaneously fit and transform data '''
        self.fit(X)  # fit first
        return self.transform(X) # return transform output

    def inverse_transform(self, X):
        ''' inverse transform data X '''
        zscore = invTanhScaler(self.a, self.b, X)  # inverse tanh
        return self.standard.inverse_transform(zscore)  # inverse standard scaler