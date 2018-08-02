# -*- encoding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import pydotplus
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

conn = psycopg2.connect(host='localhost', database='home_credit_default_risk',
                        user='postgres', password='postgres')
cur = conn.cursor()

sql = 'select * from application_train'

df = pd.read_sql(sql=sql, con=conn)
# df = pd.read_csv('../data/application_train.csv')

X = df.iloc[:, 2:]
y = df['target']

X = X[['cnt_children', 'amt_income_total', 'days_birth', 'days_employed', 'days_registration',
       'days_id_publish']].fillna(0)
X = X.astype(float)

# for column in df.columns:
#     print(column)
#     print(df.loc[0, column])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

print(X_train.index[:3])
print(y_train.index[:3])

# RandomForest
# print('Random Forest')
# rf = RandomForestClassifier(random_state=0)
# rf.fit(X_train, y_train)

# 特徴量と重要度を取得
# rf_features = X_train.columns
# rf_importances = rf.feature_importances_
# rf_indices = np.argsort(rf_importances)

# 特徴量と重要度を可視化
# plt.figure(figsize=(15,6))
# plt.barh(range(len(rf_indices)), rf_importances[rf_indices], color='b', align='center')
# plt.yticks(range(len(rf_indices)), rf_features[rf_indices])
# plt.show()

# rf_score_train = rf.score(X_train, y_train)
# rf_score_test = rf.score(X_test, y_test)

# print(rf_score_train, rf_score_test)

# 決定木
print('決定木')
dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train, y_train)

# 可視化してPDF出力
# dot_data = StringIO()
# export_graphviz(dt, out_file=dot_data)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# dot_data = export_graphviz(dt)
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_png('tree_180802.png')
# with open('tree.dot', mode='w') as f:
#     export_graphviz(dt, out_file=f)

dt_score_train = dt.score(X_train, y_train)
dt_score_test = dt.score(X_test, y_test)

print(dt_score_train, dt_score_test)

""" 
分岐条件や閾値を出力する 
参考URL
https://own-search-and-study.xyz/2016/12/25/scikit-learn%E3%81%A7%E5%AD%A6%E7%BF%92%E3%81%97%E3%81%9F%E6%B1%BA%E5%AE%9A%E6%9C%A8%E6%A7%8B%E9%80%A0%E3%81%AE%E5%8F%96%E5%BE%97%E6%96%B9%E6%B3%95%E3%81%BE%E3%81%A8%E3%82%81/
"""
# 分岐条件の変数名（IDと名称の対応はfeature_namesがもっている）
print(dt.tree_.feature)
# 分岐条件の閾値（scikit-learnでは不等号は常に≦で扱われる）
print(dt.tree_.threshold)
# 到達データの偏り度（指標は学習時に指定されたもの）
print(dt.tree_.impurity)
# 到達データ総数
print(dt.tree_.n_node_samples)
# クラス別到達データ数
print(dt.tree_.value)
# 各ノードの親子関係（leftは左側の子ノード、rightは右側の子ノードID）
print(dt.tree_.children_left)
print(dt.tree_.children_right)

print('finish!')
