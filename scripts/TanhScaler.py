# -*- coding: utf-8 -*-
"""
Created on Thu Jun 06 01:11:08 2018

@author: Nicholas
"""

from __future__ import division, print_function
import numpy as np
from sklearn.preprocessing import StandardScaler


# tanh function with range (0, 1)
tanhScaler = lambda x: 0.5*(np.tanh(x)+1)

# tanh scaler class
class TanhScaler:
    ''' this scaler feeds the z-scre from the standard scaler into a tanh function

        the tanh function allows for the output to be less sensitive to outliers and maps
        all features to a common numerical domain '''

    def __init__(self):
        ''' initialize standard scaler '''
        self.standard = StandardScaler()

    def fit(self, X):
        ''' fit standard scaler to data X '''
        self.standard.fit(X)

    def transform(self, X):
        ''' transform data X '''
        zscore = self.standard.transform(X)  # tranform with standard scaler first
        return tanhScaler(zscore)  # return tanh scaled data

    def fit_transform(self, X):
        ''' simultaneously fit and transform data '''
        self.fit(X)  # fit first
        return self.transform(X) # return transform output
