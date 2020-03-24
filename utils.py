# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 07:39:15 2020

@author: Andrei
"""
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import namedtuple

#%matplotlib inline

def set_pretty_prints():
  pd.set_option('display.max_rows', 500)
  pd.set_option('display.max_columns', 500)
  pd.set_option('display.max_colwidth', 500)
  pd.set_option('display.width', 1000)
  pd.set_option('precision', 4)    
  np.set_printoptions(precision=2)
  np.set_printoptions(suppress=True)
  
  
  
def load_dataset(dataset_name, dev_slice=0.1):
  func_name = 'load_'+dataset_name
  if not hasattr(datasets, func_name):
    raise ValueError("Unknown dataset '{}'".format(func_name))
  func = getattr(datasets, func_name)
  obj = func()
  df = pd.DataFrame(obj.data, columns=obj.feature_names)
  
  x_trn, x_ttt, y_trn, y_ttt = train_test_split(obj.data, obj.target, test_size=dev_slice * 2)
  
  x_dev, x_test, y_dev, y_test = train_test_split(x_ttt, y_ttt, test_size=0.5)
  
  dct = {
      'x_train' : x_trn,
      'y_train' : y_trn,
      'x_dev'   : x_dev,
      'y_dev'   : y_dev,
      'x_test'  : x_test,
      'y_test'  : y_test,
      'df'      : df,
      'desc'    : obj.DESCR,
      }
  Dataset = namedtuple('Dataset', list(dct.keys()))
  return Dataset(**dct)

def fetch_dataset(dataset_name, dev_slice=0.1):
  func_name = 'fetch_'+dataset_name
  if not hasattr(datasets, func_name):
    raise ValueError("Unknown dataset '{}'".format(func_name))
  func = getattr(datasets, func_name)
  obj = func(data_home='datasets/'+dataset_name)
  df = pd.DataFrame(obj.data, columns=obj.feature_names)
  
  x_trn, x_ttt, y_trn, y_ttt = train_test_split(obj.data, obj.target, test_size=dev_slice * 2)
  
  x_dev, x_test, y_dev, y_test = train_test_split(x_ttt, y_ttt, test_size=0.5)
  
  dct = {
      'x_train' : x_trn,
      'y_train' : y_trn,
      'x_dev'   : x_dev,
      'y_dev'   : y_dev,
      'x_test'  : x_test,
      'y_test'  : y_test,
      'df'      : df,
      'desc'    : obj.DESCR,
      }
  Dataset = namedtuple('Dataset', list(dct.keys()))
  return Dataset(**dct)
  

if __name__ == '__main__':
  boston = load_dataset('boston')
  calif = fetch_dataset('california_housing')
  diabt = load_dataset('diabetes')
  bc = load_dataset('breast_cancer')
  
  