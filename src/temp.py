import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from _collections import defaultdict
import statsmodels.api as sm
from sklearn.metrics import fbeta_score
from itertools import compress
from sklearn.ensemble.forest import RandomForestClassifier
import sys
from src import image_transforms as it
    

def tt():
    
    items = []
    
    df2 = pd.read_csv("C:/Users/Prasad/Desktop.rejig.csv")
    colNames = ["image_name"]
    colNamesDict = {}
    for i in range(len(colNames)):
        colNamesDict[i] = colNames[i]
    df2 = df2.rename(columns=colNamesDict)
    df_train_sample = pd.merge(df_train_sample, df2, on="image_name")
    return df_train_sample