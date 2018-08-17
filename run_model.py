# -*- encoding:utf-8 -*-
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pydotplus
import graphviz
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import precision_recall_curve, auc, roc_curve


def main():

    # """ データ加工後の結果を説明変数と目的変数に分割する """
    # input_csv_processed = r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_all\df_res.csv'
    # df = pd.read_csv(input_csv_processed, index_col=None)
    # x = df.drop(columns=['a_id', 'a_main_dis_code1', 'a_main_dis_code2', 'count_cer_a',
    #                      'b_id', 'b_main_dis_code1', 'b_main_dis_code2', 'count_cer_b',
    #                      'a_tol_flg'], axis=1)
    # y = df['a_tol_flg']
    # x.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\x_org.csv')
    # y.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\y_org.csv')
    # exit()

    """ 説明変数と目的変数を読み込む """
    csv_name = '4_2'  # CSVファイル名を設定
    input_csv_x = r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\x_{0}.csv'.format(csv_name)
    input_csv_y = r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\y_{0}.csv'.format(csv_name)
    x = pd.read_csv(input_csv_x, index_col=0)
    y = pd.read_csv(input_csv_y, index_col=0)
    y.columns = ['y']  # TODO 手動でカラム名をyにする必要あり注意
    print('csv読み込み完了')

    # """ モデルの作成単位に分割する """
    # value = 0.5
    # x_yes = x[(x['same_amain_bmain'] > value)]
    # # x_yes = x[(x['same_acause_bmain'] > value) | (x['same_amain_bcause'] > value) | (x['same_acause_bcause'] > value)]
    # # x_yes = x[(x['chu_main_190-199_now'] > value) & (x['chu_main_190-199_bef'] > value)]
    # y_yes = y.loc[x_yes.index, :]
    # # x_yes1 = x[((x['chu_main_190-199_now'] <= value) | (x['chu_main_190-199_bef'] <= value)) &
    # #            (x['chu_main_190-199_now'] > value) & (x['chu_main_150-159_bef'] > value)]
    # # y_yes1 = y.loc[x_yes1.index, :]
    # # x_yes2 = x[((x['chu_main_190-199_now'] <= value) | (x['chu_main_190-199_bef'] <= value)) &
    # #            ((x['chu_main_190-199_now'] <= value) | (x['chu_main_150-159_bef'] <= value)) &
    # #            (x['chu_main_150-159_now'] > value) & (x['chu_main_190-199_bef'] > value)]
    # # y_yes2 = y.loc[x_yes2.index, :]
    # # x_yes3 = x[((x['chu_main_190-199_now'] <= value) | (x['chu_main_190-199_bef'] <= value)) &
    # #            ((x['chu_main_190-199_now'] <= value) | (x['chu_main_150-159_bef'] <= value)) &
    # #            ((x['chu_main_150-159_now'] <= value) | (x['chu_main_190-199_bef'] <= value)) &
    # #            (x['chu_main_150-159_now'] > value) & (x['chu_main_150-159_bef'] > value)]
    # # y_yes3 = y.loc[x_yes3.index, :]
    # x_no = x[(x['same_amain_bmain'] <= value)]
    # # x_no = x[(x['same_acause_bmain'] <= value) & (x['same_amain_bcause'] <= value) & (x['same_acause_bcause'] <= value)]
    # # x_no = x[((x['chu_main_190-199_now'] <= value) | (x['chu_main_190-199_bef'] <= value)) &
    # #          ((x['chu_main_190-199_now'] <= value) | (x['chu_main_150-159_bef'] <= value)) &
    # #          ((x['chu_main_150-159_now'] <= value) | (x['chu_main_190-199_bef'] <= value)) &
    # #          ((x['chu_main_150-159_now'] <= value) | (x['chu_main_150-159_bef'] <= value))]
    # y_no = y.loc[x_no.index, :]
    # # x_yes = drop_all0_columns(x_yes)
    # # x_yes1 = drop_all0_columns(x_yes1)
    # # x_yes2 = drop_all0_columns(x_yes2)
    # # x_yes3 = drop_all0_columns(x_yes3)
    # x_no = drop_all0_columns(x_no)
    # y_yes.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\y_yes.csv')
    # # y_yes1.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\y_yes1.csv')
    # # y_yes2.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\y_yes2.csv')
    # # y_yes3.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\y_yes3.csv')
    # y_no.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\y_no.csv')
    # x_yes.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\x_yes.csv')
    # # x_yes1.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\x_yes1.csv')
    # # x_yes2.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\x_yes2.csv')
    # # x_yes3.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\x_yes3.csv')
    # x_no.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\x_no.csv')
    # exit()

    print('説明変数と目的変数のサイズチェック', x.shape, y.shape)

    y_tol = y[y['y'] == 1]
    y_notol = y[y['y'] == 0]
    x_tol = x.loc[y_tol.index, :]
    x_notol = x.loc[y_notol.index, :]
    x_tol_train, x_tol_test, y_tol_train, y_tol_test = train_test_split(x_tol, y_tol, random_state=1)
    x_notol_train, x_notol_test, y_notol_train, y_notol_test = train_test_split(x_notol, y_notol, random_state=1)
    x_train, y_train = pd.concat([x_tol_train, x_notol_train], axis=0), pd.concat([y_tol_train, y_notol_train])
    x_test, y_test = pd.concat([x_tol_test, x_notol_test], axis=0), pd.concat([y_tol_test, y_notol_test])

    print(datetime.today(), '学習データと評価データに分割完了')
    print('学習データと評価データのサイズチェック', x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # clf = RandomForestClassifier(random_state=0)
    # clf = RandomForestClassifier(random_state=0, n_estimators=100, min_samples_leaf=10)
    clf = RandomForestClassifier(random_state=0, n_estimators=100, min_samples_leaf=1)
    # clf = RandomForestClassifier(random_state=0, n_estimators=300)

    clf.fit(x_train, y_train)
    joblib.dump(clf, r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\rf_all_variables.sav')
    print('モデル構築＆保存完了')

    # 精度を評価する
    evaluate_model(clf, x_train, y_train, x_test,  y_test, 'rf')

    print(datetime.today())

    # # グリッドサーチの場合
    # params = {'n_estimators': [3, 10, 30, 100, 300, 1000], 'n_jobs': [-1]}
    # scores = ['accuracy', 'precision', 'recall']
    # mod = RandomForestClassifier()
    # for score in scores:
    #     print('score')
    #     cv = GridSearchCV(mod, params, cv=5, scoring=score)
    #     clf.fit(x_train, y_train)
    #     print(clf.best_estimator_)
    #     for params, mean_score, all_score in clf.grid_scores_:
    #         print(mean_score, all_score, params)

    # 重要度を出力し、上位100個の変数を取り出す
    select_columns = output_importances(x_test, clf, 100)

    """ 再学習 """
    print('\n\n\n再学習')

    # 重要度が上位の項目に絞る
    x_train = x_train[select_columns]
    x_test = x_test[select_columns]
    # x_train.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\x_train.csv')
    # x_test.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\x_test.csv')
    # y_train.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\y_train.csv')
    # y_test.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\y_test.csv')
    # # # x_train = pd.read_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\x_train.csv', index_col=0)
    # # # x_test = pd.read_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\x_test.csv', index_col=0)
    # # # y_train = pd.read_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\y_train.csv', index_col=0)
    # # # y_test = pd.read_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\y_test.csv', index_col=0)
    # #
    print('説明変数と目的変数のサイズチェック', x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    clf = RandomForestClassifier(random_state=0, n_estimators=1000, min_samples_leaf=1)

    clf.fit(x_train, y_train)
    joblib.dump(clf, r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\rf_part_variables.csv')
    print('再学習モデル構築＆保存完了')

    # 精度を評価する
    evaluate_model(clf, x_train, y_train, x_test, y_test, 'rf')

    # 重要度を出力する
    output_importances(x_test, clf, 0)

    print(datetime.today())
    """ 再学習おわり """
    #
    # # 決定木
    # print('\n\n\n決定木')
    # # dt = DecisionTreeClassifier(random_state=0, max_leaf_nodes=10)
    # # dt = DecisionTreeClassifier(random_state=0, min_samples_leaf=4)
    # dt = DecisionTreeClassifier(random_state=0, max_depth=7, min_samples_leaf=3)
    # # dt = DecisionTreeClassifier(random_state=0)
    #
    # dt.fit(x_train, y_train)
    # joblib.dump(dt, r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\dt_variables.csv')
    # print('決定木モデル構築＆保存完了')
    #
    # # 精度を評価する
    # evaluate_model(dt, x_train, y_train, x_test, y_test, 'dt')
    #
    # # 可視化してPDF出力
    # output_tree(dt, x_test)


    """
    分岐条件や閾値を出力する
    参考URL
    https://own-search-and-study.xyz/2016/12/25/scikit-learn%E3%81%A7%E5%AD%A6%E7%BF%92%E3%81%97%E3%81%9F%E6%B1%BA%E5%AE%9A%E6%9C%A8%E6%A7%8B%E9%80%A0%E3%81%AE%E5%8F%96%E5%BE%97%E6%96%B9%E6%B3%95%E3%81%BE%E3%81%A8%E3%82%81/
    """

    # # 分岐条件の変数名（IDと名称の対応は学習データの順番？）
    # # -2は分岐条件がない（リーフである）ことを示す
    # print('分岐条件の変数名')
    # print(dt.tree_.feature)
    # # 分岐条件の閾値（scikit-learnでは不等号は常に≦で扱われる＆Trueが左下に進む）
    # # -2は分岐条件がない（リーフである）ことを示す
    # print('分岐条件の閾値')
    # print(dt.tree_.threshold)
    # # 到達データの偏り度（指標は学習時に指定されたginiやinfoGain）
    # print('到達データの偏り度')
    # print(dt.tree_.impurity)
    # # 到達データ総数
    # print('到達データ総数')
    # print(dt.tree_.n_node_samples)
    # # クラス別到達データ数
    # print('クラス別到達データ数')
    # print(dt.tree_.value)
    # # 各ノードの親子関係（leftは左側の子ノード、rightは右側の子ノードID）※0スタート
    # print('各ノードの親子関係')
    # print(dt.tree_.children_left)
    # print(dt.tree_.children_right)
    #
    # result = get_result_dt(dt)
    #
    # print('決定木をテーブルにまとめる')
    # print(result)
    # print(result.shape)

    # path_train = dt.apply(x_train)
    # path_test = dt.apply(x_test)
    # path_train = pd.DataFrame(path_train)
    # path_test = pd.DataFrame(path_test)
    # path_train.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\path_train.csv')
    # path_test.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\path_test.csv')


def evaluate_model(model, x_train, y_train, x_test, y_test, model_name):

    pred_train = model.predict(x_train)
    proba_train = model.predict_proba(x_train)
    pred_test = model.predict(x_test)
    proba_test = model.predict_proba(x_test)

    pd.DataFrame(proba_train).to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\predict_proba_train_{0}.csv'.format(model_name))
    pd.DataFrame(proba_test).to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\predict_proba_test_{0}.csv'.format(model_name))

    print('confusion matrix train 左上：共に非通算、右上：実際非通算 予測通算、左下：実際通算 予測非通算、右下：共に通算')
    print(confusion_matrix(y_train, pred_train))
    print('confusion matrix train 左上：共に非通算、右上：実際非通算 予測通算、左下：実際通算 予測非通算、右下：共に通算')
    print(confusion_matrix(y_test, pred_test))
    print('')
    print('train 適合率（予測正のうち実際正の割合）、再現率（実際正のうち予測正の割合）、F値（適合率と再現率の調和平均）')
    print(classification_report(y_train, pred_train, target_names=['非通算', '通算'], digits=5))
    print('test 適合率（予測正のうち実際正の割合）、再現率（実際正のうち予測正の割合）、F値（適合率と再現率の調和平均）')
    print(classification_report(y_test, pred_test, target_names=['非通算', '通算'], digits=5))
    print('')

    precision_tr, recall_tr, thresholds_tr = precision_recall_curve(y_train, proba_train[:, 1])
    precision_ts, recall_ts, thresholds_ts = precision_recall_curve(y_test, proba_test[:, 1])
    area_tr = auc(recall_tr, precision_tr)
    area_ts = auc(recall_ts, precision_ts)
    print('Area Under Curve:{0}'.format(round(area_tr, 3)))
    print('Area Under Curve:{0}'.format(round(area_ts, 3)))
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(recall_tr, precision_tr, color='b', label='train (area={0})'.format(round(area_tr, 3)))
    ax1.plot(recall_ts, precision_ts, color='r', label='test (area={0})'.format(round(area_ts, 3)))
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlim([0.0, 1.0])
    ax1.set_title('Precision-Recall')
    ax1.legend(loc='lower left')

    fpr_tr, tpr_tr, thresholds_tr = roc_curve(y_train, proba_train[:, 1])
    fpr_ts, tpr_ts, thresholds_ts = roc_curve(y_test, proba_test[:, 1])
    roc_auc_tr = auc(fpr_tr, tpr_tr)
    roc_auc_ts = auc(fpr_ts, tpr_ts)
    print('Area under the ROC curve:{0}'.format(round(roc_auc_tr, 3)))
    print('Area under the ROC curve:{0}'.format(round(roc_auc_ts, 3)))
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(fpr_tr, tpr_tr, color='b', label='train (area={0})'.format(round(roc_auc_tr, 3)))
    ax2.plot(fpr_ts, tpr_ts, color='r', label='test (area={0})'.format(round(roc_auc_ts, 3)))
    ax2.set_xlabel('False Positivie Rate: 1-Recall of nototal')
    ax2.set_ylabel('True Positivie Rate: Recall of total')
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlim([0.0, 1.0])
    ax2.set_title('Receiver operating characteristic example')
    ax2.legend(loc='lower left')

    plt.savefig(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\AUCandROC_{0}.png'.format(model_name))

    return None


def output_importances(x, clf, select_num):
    # 特徴量の重要度を取得する
    feature_imps = clf.feature_importances_
    print(len(feature_imps), x.shape)
    # 特徴量の名前
    label = x.columns[0:]
    # 必要な項目抽出用
    select_columns = []
    # 特徴量の重要度順（降順）
    indices = np.argsort(feature_imps)[::-1]
    for i in range(len(feature_imps)):
        print(str(i + 1) + "   " + str(label[indices[i]]) + "   " + str(feature_imps[indices[i]]))

        # 上位100の変数を取り出す
        if i < select_num:
            select_columns.append(str(label[indices[i]]))

    return select_columns


def output_tree(dt, x):
    dot_data = tree.export_graphviz(dt,  # 決定木オブジェクトを一つ指定する
                                    out_file=None,  # ファイルは介さずにGraphvizにdot言語データを渡すのでNone
                                    filled=True,  # Trueにすると、分岐の際にどちらのノードに多く分類されたのか色で示してくれる
                                    rounded=True,  # Trueにすると、ノードの角を丸く描画する。
                                    feature_names=x.columns,  # これを指定しないとチャート上で特徴量の名前が表示されない
                                    class_names=['non_tol', 'tol'],  # これを指定しないとチャート上で分類名が表示されない
                                    special_characters=True  # 特殊文字を扱えるようにする
                                    )
    graph = pydotplus.graphviz.graph_from_dot_data(dot_data)
    graph.write_png(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\to_group_new\tree.png')


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


def drop_all0_columns(df):
    # 値がすべて0のカラムを削除する
    drop_columns = []
    for column in df.columns:
        if df[column].max() == 0:
            print(column)
            drop_columns.append(column)
    print('値がすべて0のカラム抽出完了。削除に入る')
    df = df.drop(drop_columns, axis=1)

    return df


if __name__ == "__main__":

    print(datetime.today(), 'START')

    main()

    print(datetime.today(), 'END')





