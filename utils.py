# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 07:39:15 2020

@author: Andrei
"""
import pandas as pd
import numpy as np

#%matplotlib inline

def set_pretty_prints():
  pd.set_option('display.max_rows', 500)
  pd.set_option('display.max_columns', 500)
  pd.set_option('display.max_colwidth', 500)
  pd.set_option('display.width', 1000)
  pd.set_option('precision', 4)    
  np.set_printoptions(precision=2)
  np.set_printoptions(suppress=True)
  
  