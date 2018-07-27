# -*- encoding: utf-8 -*-
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('../data/application_train.csv')
X = df.iloc[:, 2:]
y = df['TARGET']

X = X[['CNT_CHILDREN', 'AMT_INCOME_TOTAL']]

# for column in df.columns:
#     print(column)
#     print(df.loc[0, column])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

print(X_train.index[:3])
print(y_train.index[:3])

clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)

score_train = clf.score(X_train, y_train)
score_test = clf.score(X_test, y_test)

print(score_train, score_test)


