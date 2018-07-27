# -*- encoding: utf-8 -*-
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('../data/application_train.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

clf = RandomForestClassifier(random_state=0)
clf.fit(X, y)

print(df.head())



