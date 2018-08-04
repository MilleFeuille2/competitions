# -*- encoding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import psycopg2
import pydotplus
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def main():
    conn = psycopg2.connect(host='localhost', database='home_credit_default_risk',
                            user='postgres', password='postgres')
    cur = conn.cursor()

    sql = 'select * from application_train'

    df = pd.read_sql(sql=sql, con=conn)
    # df = pd.read_csv('../data/application_train.csv')

    X = df.iloc[:, 2:]
    y = df['target']

    # X = X[['cnt_children', 'amt_income_total', 'days_birth',
    #        'days_employed', 'days_registration','days_id_publish']].fillna(0)
    X = X[['cnt_children', 'amt_income_total']].fillna(0)
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
    dt = DecisionTreeClassifier(random_state=0,
                                max_depth=9,
                                min_samples_leaf=500)
    dt.fit(X_train, y_train)

    # 可視化してPDF出力
    # dot_data = StringIO()
    # export_graphviz(dt, out_file='../output/tree.dot')
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # dot_data = export_graphviz(dt)
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # graph.write_png('tree_180802.png')
    # with open('../output/tree.dot', mode='w') as f:
    #     export_graphviz(dt, out_file=f)

    dt_score_train = dt.score(X_train, y_train)
    dt_score_test = dt.score(X_test, y_test)

    print(dt_score_train, dt_score_test)

    """ 
    分岐条件や閾値を出力する 
    参考URL
    https://own-search-and-study.xyz/2016/12/25/scikit-learn%E3%81%A7%E5%AD%A6%E7%BF%92%E3%81%97%E3%81%9F%E6%B1%BA%E5%AE%9A%E6%9C%A8%E6%A7%8B%E9%80%A0%E3%81%AE%E5%8F%96%E5%BE%97%E6%96%B9%E6%B3%95%E3%81%BE%E3%81%A8%E3%82%81/
    """

    # 分岐条件の変数名（IDと名称の対応は学習データの順番？）
    # -2は分岐条件がない（リーフである）ことを示す
    print('分岐条件の変数名')
    print(dt.tree_.feature)
    # 分岐条件の閾値（scikit-learnでは不等号は常に≦で扱われる＆Trueが左下に進む）
    # -2は分岐条件がない（リーフである）ことを示す
    print('分岐条件の閾値')
    print(dt.tree_.threshold)
    # 到達データの偏り度（指標は学習時に指定されたginiやinfoGain）
    print('到達データの偏り度')
    print(dt.tree_.impurity)
    # 到達データ総数
    print('到達データ総数')
    print(dt.tree_.n_node_samples)
    # クラス別到達データ数
    print('クラス別到達データ数')
    print(dt.tree_.value)
    # 各ノードの親子関係（leftは左側の子ノード、rightは右側の子ノードID）※0スタート
    print('各ノードの親子関係')
    print(dt.tree_.children_left)
    print(dt.tree_.children_right)

    result = get_result_dt(dt)

    print('決定木をテーブルにまとめる')
    print(result)
    print(result.shape)


def get_result_dt(dt):
    left = dt.tree_.children_left
    right = dt.tree_.children_right
    get_left = []
    result_row = np.full(dt.max_depth + 1, -1)

    i, j = 0, 0  # 親、子供

    result_row[0] = i  # 項目nにiを登録
    j = left[i]  # jに左の子[i]を登録

    n = 1  # 登録する深さ（0は一番上）

    ''' 1レコード目 '''
    # j（左の子）が存在する場合
    while j != -1:

        get_left.append(i)  # 左側取得リストに親を登録
        result_row[n] = j  # 項目nに子を登録

        # 親と子を1深くする
        i, j, n = one_depth(i, j, n, left)

    # 結果格納用変数を作成する
    result = result_row

    ''' 2レコード目以降 '''
    # 左側取得リストの中身がある場合
    while len(get_left) > 0:

        i = get_left.pop(-1)  # 左側取得リストの最後の要素を親とする（リストからは削除）
        n = int(np.where(result_row == i)[0]) + 1  # 親の深さ+1を取得
        j = right[i]  # 右の子を取得
        result_row[n] = j  # 項目nにjを登録

        # 親と子を1深くする
        i, j, n = one_depth(i, j, n, left)

        # j（左の子）が存在する場合
        while j != -1:
            get_left.append(i)  # 左側取得リストに親を登録
            result_row[n] = j  # 項目nに子を登録

            # 親と子を1深くする
            i, j, n = one_depth(i, j, n, left)

        # 結果用変数に登録する
        result_row[n:] = np.full(len(result_row[n:]), -1)  # 以降の項目を-1に更新
        result = np.append(result, result_row)

    # 結果用変数を二次元にする
    result = result.reshape(-1, dt.max_depth+1)

    return result


def one_depth(i, j, n, left):
    # 親と（左の）子を1深くし、nをインクリメントする
    return j, left[j], n + 1


if __name__ == '__main__':

    main()

    print('finish!')
