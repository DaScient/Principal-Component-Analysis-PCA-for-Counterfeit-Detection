# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:09:59 2017

@author: don
"""
import csv
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd


df=pd.read_csv('arg-ld_tv.csv', sep=',',header=1, usecols = range(5,85))
df.values
            
#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])


pca = PCA(n_components=2)

pca.fit(df)

PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)

print(pca.explained_variance_ratio_)
#[ 0.99244...  0.00755...]



pca = PCA(n_components=2, svd_solver='full')

pca.fit(df)  

PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
  svd_solver='full', tol=0.0, whiten=False)

print(pca.explained_variance_ratio_) 
#[ 0.99244...  0.00755...]



pca = PCA(n_components=1, svd_solver='arpack')

pca.fit(df)

PCA(copy=True, iterated_power='auto', n_components=1, random_state=None,
  svd_solver='arpack', tol=0.0, whiten=False)

print(pca.explained_variance_ratio_) 
#[ 0.99244...]



