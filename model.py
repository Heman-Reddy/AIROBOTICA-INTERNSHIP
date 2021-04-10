# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 10:35:56 2021

@author: heman
"""

import numpy as np
import pandas as pd
import pickle
df=pd.read_csv("framingham.csv")
df=df.rename(columns={"male":"Gender"})
df=df.drop("education",axis=1)
df=round(df)
df.cigsPerDay=df.cigsPerDay.fillna(method="ffill")
df.BPMeds=df.BPMeds.fillna(method="ffill")
df.totChol=df.totChol.fillna(method="ffill")
df.BMI=df.BMI.fillna(method="ffill")
df.TenYearCHD=df.TenYearCHD.fillna(method="ffill")
df.glucose=df.glucose.fillna(method="ffill")
df.heartRate=df.heartRate.fillna(method="ffill")
X=df.iloc[:,:-1]
y=df.iloc[:,-1:]
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X,y)
pickle.dump(lr, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
