# -*- encoding:utf-8 -*-
import csv
import os
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
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV


def main():
    #
    # """ データ加工後の結果を説明変数と目的変数に分割する """
    # input_csv_processed = r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_all\df_res.csv'
    # df = pd.read_csv(input_csv_processed, index_col=None)
    # df = df[['a_id', 'a_main_dis_code1', 'a_main_dis_code2', 'count_cer_a',
    #          'b_id', 'b_main_dis_code1', 'b_main_dis_code2', 'count_cer_b', 'a_tol_flg']]
    # df.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\xy_org.csv')
    # print('xy出力完了')
    # x = df.drop(columns=['a_id', 'a_main_dis_code1', 'a_main_dis_code2', 'count_cer_a',
    #                      'b_id', 'b_main_dis_code1', 'b_main_dis_code2', 'count_cer_b', 'a_tol_flg'], axis=1)
    # y = df['a_tol_flg']
    # y.columns = ['y']
    # x.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\x_org.csv')
    # y.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\y_org.csv')
    # exit()
    #
    """ 説明変数と目的変数を読み込む """
    csv_name = 'org'  # CSVファイル名を設定
    input_csv_x = r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\x_{0}.csv'.format(csv_name)
    input_csv_y = r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\y_{0}.csv'.format(csv_name)
    x = pd.read_csv(input_csv_x, index_col=0)
    y = pd.read_csv(input_csv_y, index_col=0)
    columns = x.columns
    class_names = ['non_tol', 'tol']
    print('csv読み込み完了')

    # 期間の日数に関する項目を削除する TODO 退院日からのminを残す場合も試す
    x = x.drop(columns=['byo_date_from_bnyu_max']) if 'byo_date_from_bnyu_max' in x.columns else x
    x = x.drop(columns=['byo_date_from_bnyu_min']) if 'byo_date_from_bnyu_min' in x.columns else x
    x = x.drop(columns=['byo_date_from_btai_max']) if 'byo_date_from_btai_max' in x.columns else x
    # x = x.drop(columns=['byo_date_from_btai_min']) if 'byo_date_from_btai_min' in x.columns else x
    x = x.drop(columns=['syosin_date_from_bnyu_max']) if 'syosin_date_from_bnyu_max' in x.columns else x
    x = x.drop(columns=['syosin_date_from_bnyu_min']) if 'syosin_date_from_bnyu_min' in x.columns else x
    x = x.drop(columns=['syosin_date_from_btai_max']) if 'syosin_date_from_btai_max' in x.columns else x
    # x = x.drop(columns=['syosin_date_from_btai_min']) if 'syosin_date_from_btai_min' in x.columns else x
    x = x.drop(columns=['gankaku_date_from_bnyu_max']) if 'gankaku_date_from_bnyu_max' in x.columns else x
    x = x.drop(columns=['gankaku_date_from_bnyu_min']) if 'gankaku_date_from_bnyu_min' in x.columns else x
    x = x.drop(columns=['gankaku_date_from_btai_max']) if 'gankaku_date_from_btai_max' in x.columns else x
    # x = x.drop(columns=['gankaku_date_from_btai_min']) if 'gankaku_date_from_btai_min' in x.columns else x
    x = x.drop(columns=['hokagankaku_date_from_bnyu_max']) if 'hokagankaku_date_from_bnyu_max' in x.columns else x
    x = x.drop(columns=['hokagankaku_date_from_bnyu_min']) if 'hokagankaku_date_from_bnyu_min' in x.columns else x
    x = x.drop(columns=['hokagankaku_date_from_btai_max']) if 'hokagankaku_date_from_btai_max' in x.columns else x
    # x = x.drop(columns=['hokagankaku_date_from_btai_min']) if 'hokagankaku_date_from_btai_min' in x.columns else x

    # # """ モデルの作成単位に分割する """
    # value = 0.5
    # x_yes1 = x[(x['same_amain_bcause'] > value) & (x['same_amain_bcause'] > value)]
    # x_org = x[(x['same_amain_bcause'] <= value) | (x['same_amain_bcause'] <= value)]
    # y_yes1 = y.loc[x_yes1.index, :]
    # y_org = y.loc[x_org.index, :]
    #
    # # x_yes2 = x_org[(x_org['chu_main_190_199_now'] > value) & (x_org['chu_main_150_159_bef'] > value)]
    # # x_org = x_org[(x_org['chu_main_190_199_now'] <= value) | (x_org['chu_main_150_159_bef'] <= value)]
    # # y_yes2 = y_org.loc[x_yes2.index, :]
    # # y_org = y_org.loc[x_org.index, :]
    # #
    # # x_yes3 = x_org[(x_org['chu_main_150_159_now'] > value) & (x_org['chu_main_190_199_bef'] > value)]
    # # x_org = x_org[(x_org['chu_main_150_159_now'] <= value) | (x_org['chu_main_190_199_bef'] <= value)]
    # # y_yes3 = y_org.loc[x_yes3.index, :]
    # # y_org = y_org.loc[x_org.index, :]
    # #
    # # x_yes4 = x_org[(x_org['chu_main_150_159_now'] > value) & (x_org['chu_main_150_159_bef'] > value)]
    # # x_org = x_org[(x_org['chu_main_150_159_now'] <= value) | (x_org['chu_main_150_159_bef'] <= value)]
    # # y_yes4 = y_org.loc[x_yes4.index, :]
    # # y_org = y_org.loc[x_org.index, :]
    #
    # x_yes1 = drop_all0_columns(x_yes1)
    # # x_yes2 = drop_all0_columns(x_yes2)
    # # x_yes3 = drop_all0_columns(x_yes3)
    # # x_yes4 = drop_all0_columns(x_yes4)
    # x_org = drop_all0_columns(x_org)
    #
    # y_yes1.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\y_yes1.csv')
    # x_yes1.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\x_yes1.csv')
    # # y_yes2.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\y_yes2.csv')
    # # x_yes2.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\x_yes2.csv')
    # # y_yes3.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\y_yes3.csv')
    # # x_yes3.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\x_yes3.csv')
    # # y_yes4.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\y_yes4.csv')
    # # x_yes4.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\x_yes4.csv')
    # y_org.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\y_org2.csv')
    # x_org.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\x_org2.csv')
    # exit()

    print('説明変数と目的変数のサイズチェック', x.shape, y.shape)

    # y_tol, y_notol = y[y['y'] == 1], y[y['y'] == 0]
    # x_tol, x_notol = x.loc[y_tol.index, :], x.loc[y_notol.index, :]
    # x_tol_train, x_tol_test, y_tol_train, y_tol_test = train_test_split(x_tol, y_tol, random_state=1)
    # x_notol_train, x_notol_test, y_notol_train, y_notol_test = train_test_split(x_notol, y_notol, random_state=1)
    # x_train, y_train = pd.concat([x_tol_train, x_notol_train], axis=0), pd.concat([y_tol_train, y_notol_train])
    # x_test, y_test = pd.concat([x_tol_test, x_notol_test], axis=0), pd.concat([y_tol_test, y_notol_test])
    #
    # print(datetime.today(), '学習データと評価データに分割完了')
    # print('学習データと評価データのサイズチェック', x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    #
    # clf = RandomForestClassifier(random_state=0, n_estimators=100, min_samples_leaf=3, max_depth=30)
    # # TODO 最終的には class_weight='balanced' を追加する ← クラスごとのデータ数の不均衡を自動的に補完してくれる
    # #
    # clf.fit(x_train, y_train)
    # joblib.dump(clf, r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\rf_all_variables.sav')
    # print('モデル構築＆保存完了')
    # #
    # # 精度を評価する
    # evaluate_model(clf, 'rf', x_train, y_train, x_test,  y_test)
    #
    # print(datetime.today())
    #
    # # 重要度を出力し、上位100個の変数を取り出す
    # select_columns = output_importances(x_test, clf, 100)
    #
    # """ グリッドサーチの場合 """
    # # input_path = r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all'
    # # x_train = pd.read_csv(os.path.join(input_path, 'x_train_100.csv'), index_col=0)
    # # x_test = pd.read_csv(os.path.join(input_path, 'x_test_100.csv'), index_col=0)
    # # y_train = pd.read_csv(os.path.join(input_path, 'y_train.csv'), index_col=0)
    # # y_test = pd.read_csv(os.path.join(input_path, 'y_test.csv'), index_col=0)
    # # x, y = pd.concat([x_train, x_test]), pd.concat([y_train, y_test])
    # # x = pd.read_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\x_org.csv', index_col=0)
    # # y = pd.read_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\y_org.csv', index_col=0)
    # # print('グリッドサーチ 開始')
    # params = {'n_estimators': [100, 500],
    #           'max_depth': [9, 15, 50, 100, None],
    #           'min_samples_leaf': [3, 5, 10]
    #           }
    # # TODO          'max_features': [None]
    # # params = {'n_estimators': [100],
    # #           'max_depth': [None],
    # #           'min_samples_leaf': [3]
    # #           }
    # #
    # rf = RandomForestClassifier()
    # clf = GridSearchCV(rf, params, cv=4, n_jobs=1)
    # clf.fit(x[select_columns], y)
    # df_result = pd.DataFrame(clf.cv_results_)
    # print(df_result)
    # df_result.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\gridsearch_result_allvariables.csv')
    # exit()

    # """ 再学習 """
    # print('\n\n\n再学習')
    #
    # # 重要度が上位の項目に絞る
    # x_train, x_test = x_train[select_columns], x_test[select_columns]
    # x_train.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\x_train_100.csv', index=0)
    # x_test.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\x_test_100.csv', index=0)
    # y_train.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\y_train.csv', index=0)
    # y_test.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\y_test.csv', index=0)
    # # # x_train = pd.read_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\x_train.csv', index_col=0)
    # # # x_test = pd.read_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\x_test.csv', index_col=0)
    # # # y_train = pd.read_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\y_train.csv', index_col=0)
    # # # y_test = pd.read_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\y_test.csv', index_col=0)
    # #
    # print('説明変数と目的変数のサイズチェック', x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    # clf = RandomForestClassifier(random_state=0, n_estimators=500, min_samples_leaf=5)
    # clf.fit(x_train, y_train)
    # # joblib.dump(clf, r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\rf_part_variables.csv')
    # print('再学習モデル構築＆保存完了')
    #
    # # 精度を評価する
    # evaluate_model(clf, 'rf', x_train, y_train, x_test, y_test)
    #
    # # 重要度を出力する
    # output_importances(x_test, clf)
    #
    # print(datetime.today())
    # """ 再学習おわり """

    # 決定木
    print('\n\n\n決定木 全件で実施')
    # dt = DecisionTreeClassifier(random_state=0, min_samples_leaf=5, max_depth=8)
    dt = DecisionTreeClassifier(random_state=0, max_depth=8)

    dt.fit(x, y)
    # dt.fit(x_train, y_train)
    joblib.dump(dt, r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\dt_variables.csv')
    print('決定木モデル構築＆保存完了')

    # # 精度を評価する
    # evaluate_model(dt, 'dt', x_train, y_train, x_test, y_test)

    # 可視化してPDF出力
    output_tree(dt, x, class_names)
    # output_tree(dt, x_test, class_names)

    # 条件分岐をアウトプットする
    output_branch(dt, class_names, x, y)
    # output_branch(dt, class_names, x_train, y_train, x_test, y_test)


def evaluate_model(model, model_name, x_train, y_train, x_test=None, y_test=None):

    pred_train = model.predict(x_train)
    proba_train = model.predict_proba(x_train)
    if x_test is not None:
        pred_test = model.predict(x_test)
        proba_test = model.predict_proba(x_test)

    # pd.DataFrame(proba_train).to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\predict_proba_train_{0}.csv'.format(model_name))
    # if x_test is not None:
    #     pd.DataFrame(proba_test).to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\predict_proba_test_{0}.csv'.format(model_name))

    print('confusion matrix train 左上：共に非通算、右上：実際非通算 予測通算、左下：実際通算 予測非通算、右下：共に通算')
    print(confusion_matrix(y_train, pred_train))
    print(confusion_matrix(y_test, pred_test) if x_test is not None else '')
    print('')
    print('Precision、Recall、F1_score')
    print(classification_report(y_train, pred_train, target_names=['非通算', '通算'], digits=5))
    print(classification_report(y_test, pred_test, target_names=['非通算', '通算'], digits=5) if x_test is not None else '')
    print('')

    # 学習データのAUC ROC
    precision_tr, recall_tr, thresholds_tr = precision_recall_curve(y_train, proba_train[:, 1])
    area_tr = auc(recall_tr, precision_tr)
    print('Area Under Curve:{0}'.format(round(area_tr, 3)))
    fpr_tr, tpr_tr, thresholds_tr = roc_curve(y_train, proba_train[:, 1])
    roc_auc_tr = auc(fpr_tr, tpr_tr)
    print('Area under the ROC curve:{0}'.format(round(roc_auc_tr, 3)))
    # 評価データのAUC ROC
    if x_test is not None:
        precision_ts, recall_ts, thresholds_ts = precision_recall_curve(y_test, proba_test[:, 1])
        area_ts = auc(recall_ts, precision_ts)
        print('Area Under Curve:{0}'.format(round(area_ts, 3)))
        fpr_ts, tpr_ts, thresholds_ts = roc_curve(y_test, proba_test[:, 1])
        roc_auc_ts = auc(fpr_ts, tpr_ts)
        print('Area under the ROC curve:{0}'.format(round(roc_auc_ts, 3)))

    # # AUCを描く
    # fig = plt.figure(figsize=(10, 10))
    # ax1 = fig.add_subplot(1, 2, 1)
    # ax1.plot(recall_tr, precision_tr, color='b', label='train (area={0})'.format(round(area_tr, 3)))
    # if x_test is not None:
    #     ax1.plot(recall_ts, precision_ts, color='r', label='test (area={0})'.format(round(area_ts, 3)))
    # ax1.set_xlabel('Recall')
    # ax1.set_ylabel('Precision')
    # ax1.set_ylim([0.0, 1.05])
    # ax1.set_xlim([0.0, 1.0])
    # ax1.set_title('Precision-Recall')
    # ax1.legend(loc='lower left')

    # # ROCを描く
    # ax2 = fig.add_subplot(1, 2, 2)
    # ax2.plot(fpr_tr, tpr_tr, color='b', label='train (area={0})'.format(round(roc_auc_tr, 3)))
    # if x_test is not None:
    #     ax2.plot(fpr_ts, tpr_ts, color='r', label='test (area={0})'.format(round(roc_auc_ts, 3)))
    # ax2.set_xlabel('False Positivie Rate: 1-Recall of nototal')
    # ax2.set_ylabel('True Positivie Rate: Recall of total')
    # ax2.set_ylim([0.0, 1.05])
    # ax2.set_xlim([0.0, 1.0])
    # ax2.set_title('Receiver operating characteristic example')
    # ax2.legend(loc='lower left')
    #
    # plt.savefig(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\AUCandROC_{0}.png'.format(model_name))

    return None


def output_importances(x, clf, select_num=None):
    # 特徴量の重要度を取得する
    feature_imps = clf.feature_importances_
    print(len(feature_imps), x.shape)
    # 特徴量の名前
    label = x.columns[0:]
    # 必要な項目抽出用
    select_columns = []
    # 特徴量の重要度順（降順）
    indices = np.argsort(feature_imps)[::-1]
    for i in range(len(feature_imps), select_num):
        # 上位100の変数を出力する＆取り出す
        print(str(i + 1) + "   " + str(label[indices[i]]) + "   " + str(feature_imps[indices[i]]))
        select_columns.append(str(label[indices[i]]))

    return select_columns


def output_tree(dt, x, class_names):
    dot_data = tree.export_graphviz(dt,  # 決定木オブジェクトを一つ指定する
                                    out_file=None,  # ファイルは介さずにGraphvizにdot言語データを渡すのでNone
                                    filled=True,  # Trueにすると、分岐の際にどちらのノードに多く分類されたのか色で示してくれる
                                    rounded=True,  # Trueにすると、ノードの角を丸く描画する。
                                    feature_names=x.columns,  # これを指定しないとチャート上で特徴量の名前が表示されない
                                    class_names=class_names,  # これを指定しないとチャート上で分類名が表示されない
                                    special_characters=True  # 特殊文字を扱えるようにする
                                    )
    graph = pydotplus.graphviz.graph_from_dot_data(dot_data)
    graph.write_png(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\tree.png')


def get_result_dt(dt):
    left = dt.tree_.children_left
    right = dt.tree_.children_right
    get_left = []
    result_row = np.full(dt.max_depth + 1, -1)
    result_row_yn = np.full(dt.max_depth + 1, '   ')
    result = []  # どのノードとどのノードが結合しているかテーブルに格納用
    result_yn = []  # 各ノードのYESNO格納用
    last_node = []  # 各条件の最終ノード格納用

    i, j = 0, 0  # 親、子供

    result_row[0] = i  # 項目nにiを登録
    j = left[i]  # jに左の子[i]を登録

    n = 1  # 登録する深さ（0は一番上）

    ''' 1レコード目 '''
    # j（左の子）が存在する場合
    while j != -1:

        get_left.append(i)  # 左側取得リストに親を登録
        result_row[n] = j  # 項目nに子を登録
        result_row_yn[n-1] = 'no'

        # 親と子を1深くする
        i, j, n = one_depth(i, j, n, left)

    # 結果格納用変数を作成する
    result = np.append(result, result_row)
    result_yn = np.append(result_yn, result_row_yn)
    last_node.append(i)

    ''' 2レコード目以降 '''
    # 左側取得リストの中身がある場合
    while len(get_left) > 0:

        i = get_left.pop(-1)  # 左側取得リストの最後の要素を親とする（リストからは削除）
        n = int(np.where(result_row == i)[0]) + 1  # 親の深さ+1を取得
        j = right[i]  # 右の子を取得
        result_row[n] = j  # 項目nにjを登録
        result_row_yn[n-1] = 'yes'

        # 親と子を1深くする
        i, j, n = one_depth(i, j, n, left)

        # j（左の子）が存在する場合
        while j != -1:
            get_left.append(i)  # 左側取得リストに親を登録
            result_row[n] = j  # 項目nに子を登録
            result_row_yn[n-1] = 'no'

            # 親と子を1深くする
            i, j, n = one_depth(i, j, n, left)

        # 結果用変数に登録する
        result_row[n:] = np.full(len(result_row[n:]), -1)  # 以降の項目を-1に更新
        result_row_yn[n-1:] = np.full(len(result_row_yn[n-1:]), '')
        result = np.append(result, result_row)
        result_yn = np.append(result_yn, result_row_yn)
        last_node.append(i)

    # 結果用変数を二次元にする
    result = result.reshape(-1, dt.max_depth+1)
    result_yn = result_yn.reshape(-1, dt.max_depth+1)

    return result, result_yn, last_node


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


def output_branch(dt, class_names, x_train, y_train, x_test=None, y_test=None):
    # 分岐条件の変数名（IDと名称の対応は学習データの順番？）
    # -2は分岐条件がない（リーフである）ことを示す
    branch_name = [x_train.columns[i] if i != -2 else '' for i in dt.tree_.feature]
    # branch_name_jp = convert_jp(branch_name)

    # 到達ノードと予測クラスを取得する
    ## 学習データ
    path_train = dt.apply(x_train)
    pred_train = dt.predict(x_train)
    ## 評価データ
    if x_test is not None:
        path_test = dt.apply(x_test)
        pred_test = dt.predict(x_test)

    # 分岐条件を二次元で取得する
    result, result_yn, last_node = get_result_dt(dt)

    result_jp = []
    result_last_node = []

    for i in range(result.shape[0]):
        row_branch, row_impurity, row_samples, row_value, row_class = [], [], [], [], []

        for j, k in zip(result[i, :].astype(int), range(result_yn.shape[1])):
            if dt.tree_.threshold[j] != -2.0:
                row_branch.append('{0} の値が {1}以上である {2}'
                                  .format(branch_name[j], dt.tree_.threshold[j], result_yn[i, k]))
                row_impurity.append(round(dt.tree_.impurity[j], 3))
                row_samples.append(dt.tree_.n_node_samples[j])
                row_value.append(dt.tree_.value[j])
                row_class.append(class_names[np.argmax(dt.tree_.value[j])])
            else:
                row_branch.append('')
                row_impurity.append('')
                row_samples.append('')
                row_value.append('')
                row_class.append('')

        result_jp.append(pd.concat([row_branch, row_impurity, row_samples, row_value, row_class], axis=1))

        # 予測クラス
        row_last_node = [class_names[np.argmax(dt.tree_.value[last_node[i]])]]
        # 学習データ
        ## 最終ノードの到達数
        row_last_node.append((path_train == last_node[i]).astype(int).sum())
        ## 最終ノードのクラス別到達数
        row_last_node.append(((path_train == last_node[i]) & (pred_train == 0)).astype(int).sum())
        row_last_node.append(((path_train == last_node[i]) & (pred_train == 1)).astype(int).sum())
        ## 実際のクラス別数
        row_last_node.append(((path_train == last_node[i]) & (y[i] == 0)).astype(int).sum())
        row_last_node.append(((path_train == last_node[i]) & (y[i] == 1)).astype(int).sum())
        # row_last_node.append(((path_train == last_node[i]) & (y_train[i] == 0)).astype(int).sum())
        # row_last_node.append(((path_train == last_node[i]) & (y_train[i] == 1)).astype(int).sum())
        # 評価データ
        if x_test is not None:
            ## 最終ノードの到達数
            row_last_node.append((path_test == last_node[i]).astype(int).sum())
            ## 最終ノードのクラス別到達数
            row_last_node.append(((path_test == last_node[i]) & (pred_test == 0)).astype(int).sum())
            row_last_node.append(((path_test == last_node[i]) & (pred_test == 1)).astype(int).sum())
            ## 実際のクラス別数
            row_last_node.append(((path_test == last_node[i]) & (y_test[i] == 0)).astype(int).sum())
            row_last_node.append(((path_test == last_node[i]) & (y_test[i] == 1)).astype(int).sum())

        result_last_node.append(row_last_node)

    df_result_jp = pd.DataFrame(result_jp)
    df_result_last_node = pd.DataFrame(result_last_node)

    df_result = pd.concat([df_result_jp, df_result_last_node], axis=1)
    df_result.index = ['条件{0}'.format(i) for i in range(1, df_result_jp.shape[0] + 1)]
    columns_branch = ['項目{0}_条件'.format(i) for i in range(1, df_result_jp.shape[1] + 1)]
    columns_impurity = ['項目{0}_gini'.format(i) for i in range(1, df_result_jp.shape[1] + 1)]
    columns_samples = ['項目{0}_到達数'.format(i) for i in range(1, df_result_jp.shape[1] + 1)]
    columns_value = ['項目{0}_クラス別到達数'.format(i) for i in range(1, df_result_jp.shape[1] + 1)]
    columns_class = ['項目{0}_予測クラス'.format(i) for i in range(1, df_result_jp.shape[1] + 1)]
    columns2 = ['pred_class', 'train_samples', 'train_pred_non_tol', 'train_pred_tol',
                'train_real_non_tol', 'train_real_tol',
                'test_samples', 'test_pred_non_tol', 'test_pred_tol',
                'test_real_non_tol', 'test_real_tol'] if x_test is not None else \
               ['pred_class', 'train_samples', 'train_pred_non_tol', 'train_pred_tol',
                'train_real_non_tol', 'train_real_tol']

    df_result.columns = columns_branch + columns_impurity + columns_samples + columns_value + \
                        columns_class + columns2

    # 適合率のカラムを作成する
    ## 学習データ
    df_tmp = df_result['pred_class', 'train_samples', 'train_real_non_tol', 'train_real_tol']
    df_tmp.columns = ['pred_class', 'samples', 'real_non_tol', 'real_tol']
    df_result['train_precision'] = df_result.apply(lambda x: calc_precision(x), axis=1)
    ## 評価データ
    if x_test is not None:
        df_tmp = df_result['pred_class', 'test_samples', 'test_real_non_tol', 'test_real_tol']
        df_tmp.columns = ['pred_class', 'samples', 'real_non_tol', 'real_tol']
        df_result['test_precision'] = df_result.apply(lambda x: calc_precision(x), axis=1)

    df_result.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model\all\result_dt.csv')


def calc_precision(x):
    if x.pred_class == 'non_tol':
        return x.real_non_tol / x.samples * 100
    else:
        return x.real_tol / x.samples * 100


def convert_jp(before):
    dict_jp = {'code_gappei_682_bef_y': '過去の合併症が682',
               'code_kiou_174_bef': '過去の既往症が174',
               'code_byo_151_now': '今回の傷病が151',
               'hosyabui_exist_a': '今回の放射性部位が存在',
               'code_cause_174_bef': '過去の原因傷病が174',
               'code_gappei_576_now_x': '今回の合併症が576',
               'code_main_194_now': '今回の主傷病が194',
               'same_akiou_bkiou': '今回と過去の既往症が同じ',
               'a_kiou_exist': '今回の既往症が存在',
               }
    after = [dict_jp[i] if i != '' else '' for i in before]

    return after


if __name__ == "__main__":

    print(datetime.today(), 'START')

    main()

    print(datetime.today(), 'END')




























