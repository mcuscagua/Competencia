# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 11:32:45 2019

@author: jcuscagu
"""
import pandas as pd

data = pd.read_csv('databinarystudents.csv')
data = data.iloc[:, 1:]