# -*- encoding:utf-8 -*-
import csv
import math
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier
import convert_jp

depth = 8
# 作業フォルダの場所とCSVファイル名を指定
# place = r'6_Phase2+alpha_maxdepth{0}'.format(depth)
# place = r'5_Phase2_chukan'
place = r'5.5_Phase2_modify_maxdepth{0}'.format(depth)
# place = r'all'
# csv_name = ['org']
# csv_name = [str(i).zfill(2) for i in range(2, 34, 2)]
# csv_name = ['02', '04', '06', '08', '10', '12', '14', '16', '18', '20', '22', '24', '26', '28', '30', '32', '34', '36', '38', '40', '41']
csv_name = [str(i).zfill(2) for i in range(1, 17)]
csv_name.extend([str(i).zfill(2) for i in range(18, 23)])
# csv_name = ['38', '40']
inpath = r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\model'


def main():
    # #
    # """ データ加工後の結果を説明変数と目的変数に分割する """
    # input_csv = r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_all\df_res.csv'
    # df = pd.read_csv(input_csv, index_col=None)
    # df = df[(df['count_cer_a'] > 0) & (df['count_cer_b'] > 0)]
    # for column in df.columns:
    #     print(column)
    # df2 = df[['a_id', 'a_main_dis_code1', 'a_main_dis_code2', 'count_cer_a',
    #           'b_id', 'b_main_dis_code1', 'b_main_dis_code2', 'count_cer_b', 'a_tol_flg']]
    # df2.to_csv(os.path.join(inpath, place, 'xy_{0}.csv'.format(csv_name[0])))
    # print('xy出力完了')
    # x = df.drop(columns=['a_id', 'a_main_dis_code1', 'a_main_dis_code2', 'count_cer_a',
    #                      'b_id', 'b_main_dis_code1', 'b_main_dis_code2', 'count_cer_b', 'a_tol_flg',
    #                      'byo_date_from_bnyu_max', 'byo_date_from_bnyu_min',
    #                      'byo_date_from_btai_max', 'byo_date_from_btai_min',
    #                      'syosin_date_from_bnyu_max', 'syosin_date_from_bnyu_min',
    #                      'syosin_date_from_btai_max', 'syosin_date_from_btai_min',
    #                      'gankaku_date_from_bnyu_max', 'gankaku_date_from_bnyu_min',
    #                      'gankaku_date_from_btai_max', 'gankaku_date_from_btai_min',
    #                      'hokagankaku_date_from_bnyu_max', 'hokagankaku_date_from_bnyu_min',
    #                      'hokagankaku_date_from_btai_max', 'hokagankaku_date_from_btai_min'], axis=1)
    # y = df['a_tol_flg']
    # y.columns = ['y']
    # x.to_csv(os.path.join(inpath, place, csv_name[0], 'x_{0}.csv'.format(csv_name[0])))
    # y.to_csv(os.path.join(inpath, place, csv_name[0], 'y_{0}.csv'.format(csv_name[0])))
    # exit()

    # df_result_all = None
    importances_all = []
    evaluate_all_rf = []
    evaluate_all_dt = []

    for i in range(len(csv_name)):
        nam = csv_name[i]
        print(nam)
        # """ モデルの作成単位に分割する """
        # columns = {'01': ['chu_main_190_199_now', 'chu_main_190_199_bef'],
        #            '02': ['chu_main_150_159_now', 'chu_main_150_159_bef'],
        #            '03': 'code_main_211_bef', '04': 'chu_main_570_579_bef',
        #            '05': 'code_main_366_bef', '06': 'chu_main_190_199_now',
        #            '07': 'chu_main_200_208_now', '08': 'chu_main_160_165_now',
        #            '09': 'chu_main_179_189_now', '10': 'chu_main_430_438_now',
        #            '11': 'code_main_211_now', '12': 'chu_main_430_438_bef',
        #            '13': 'chu_main_580_589_now', '14': 'code_main_366_now',
        #            '15': 'same_amain_bmain', '16': 'same_acause_bmain'
        #            }
        # columns = {'18': 'a_byogen_exist', '19': 'chu_main_190_199_bef',
        #            '20': 'chu_main_410_414_bef', '21': 'chu_main_480_487_now'
        #            }
        # x = pd.read_csv(os.path.join(inpath, place, nam, 'x_{0}.csv'.format(nam)), index_col=0)
        # y = pd.read_csv(os.path.join(inpath, place, nam, 'y_{0}.csv'.format(nam)), index_col=0)
        # separate_data(x, y, 0.5, columns)

        """ 分析開始 """
        x = pd.read_csv(os.path.join(inpath, place, nam, 'x_{0}.csv'.format(nam)), index_col=0)
        y = pd.read_csv(os.path.join(inpath, place, nam, 'y_{0}.csv'.format(nam)), index_col=0)
        y.columns = ['y']
        columns = x.columns
        class_names = ['non_tol', 'tol']
        print('read csv Done')

        # # # # 期間の日数に関する項目を削除する
        # # x = drop_date_from_b_nyutai(x)
    
        y_tol, y_notol = y[y['y'] == 1], y[y['y'] == 0]
        x_tol, x_notol = x.loc[y_tol.index, :], x.loc[y_notol.index, :]
        x_tol_train, x_tol_test, y_tol_train, y_tol_test = train_test_split(x_tol, y_tol, random_state=1)
        x_notol_train, x_notol_test, y_notol_train, y_notol_test = train_test_split(x_notol, y_notol, random_state=1)
        x_train, y_train = pd.concat([x_tol_train, x_notol_train], axis=0), pd.concat([y_tol_train, y_notol_train])
        x_test, y_test = pd.concat([x_tol_test, x_notol_test], axis=0), pd.concat([y_tol_test, y_notol_test])
        y_test.to_csv(os.path.join(inpath, place, nam, 'y_test_{0}.csv'.format(nam)))
    
        print('shape of x and y', x.shape, y.shape)
        print(datetime.today(), 'separate train and test Done')
        print('shape of train and test', x_train.shape, y_train.shape, x_test.shape, y_test.shape)

        # """ 傷病の組み合わせを決めるためのRandom Forest """
        # # clf = RandomForestClassifier(random_state=0, n_estimators=300, min_samples_leaf=33)
        # clf = RandomForestClassifier(random_state=1, n_estimators=300, min_samples_leaf=33, class_weight='balanced',
        #                              min_impurity_decrease=0.0001)
        #
        # clf.fit(x_train, np.ravel(y_train))
        # dummy = DummyClassifier(strategy='most_frequent').fit(x_train, y_train)
        #
        # joblib.dump(clf, os.path.join(inpath, place, nam, 'rf_{0}.sav'.format(nam)))
        # # clf = joblib.load(os.path.join(inpath, place, nam, 'rf_{0}.sav'.format(nam)))
        # print(datetime.today(), 'fit and save model Done')
        # print(clf.get_params)
        #
        # # 精度を評価する
        # evaluate_model(nam, clf, 'rf3', dummy, x_train, y_train, x_test, y_test)
        #
        # # 重要度を出力し、上位50個の変数を取り出す
        # select_columns = output_importances(nam, 'rf', x_test, clf, 50)
        # exit()

        """ 重要項目を出力するためのRandom Forest """
        print('グリッドサーチ 開始')
        params = {'random_state': [1],
                  'n_estimators': [300],
                  'min_samples_leaf': [3, 7, 15, 25],  # 学習データの3/4のため最大25
                  'min_impurity_decrease': [0, 0.1, 0.01, 0.001, 0.0001, 0.00001],
                  # 'min_samples_leaf': [3, 25],  # 学習データの3/4のため最大25
                  # 'min_impurity_decrease': [0],
                  'class_weight': ['balanced', None]
                  }
        # TODO          'max_features': [auto, log2, None]
        clf = GridSearchCV(RandomForestClassifier(), params, cv=4, n_jobs=1, scoring='roc_auc')
        clf.fit(x_train, np.ravel(y_train))
        joblib.dump(clf, os.path.join(inpath, place, nam, 'rf1_{0}.sav'.format(nam)))
        # clf = joblib.load(os.path.join(inpath, place, nam, 'rf1_{0}.sav'.format(nam)))

        # グリッドサーチの結果を保存する
        df_result = pd.DataFrame(clf.cv_results_)
        df_result.\
            to_csv(os.path.join(inpath, place, nam, 'gridsearch_result_{0}_rf1.csv'.format(nam)))

        # AUCが最良のパラメータで再学習する
        print('finish grid search', clf.best_params_)
        min_samples_leaf = int(clf.best_params_['min_samples_leaf'] * 4 / 3)
        min_impurity_decrease = clf.best_params_['min_impurity_decrease']
        class_weight = clf.best_params_['class_weight']
        clf = RandomForestClassifier(random_state=1, n_estimators=300, min_samples_leaf=min_samples_leaf,
                                     min_impurity_decrease=min_impurity_decrease, class_weight=class_weight)
        clf = clf.fit(x_train, np.ravel(y_train))
        joblib.dump(clf, os.path.join(inpath, place, nam, 'rf2_{0}.sav'.format(nam)))
        # clf = joblib.load(os.path.join(inpath, place, nam, 'rf2_{0}.sav'.format(nam)))

        """ """
        pred_proba_test = np.c_[clf.predict(x_test), clf.predict_proba(x_test)[:, 1], clf.predict_proba(x_test)[:, 0]]
        df_res_rf = pd.DataFrame(pred_proba_test, index=y_test.index, columns=['pred', 'proba_tol', 'proba_notol'])
        df_res_rf['y'] = y_test['y']
        df_res_rf['groupno'] = [int(nam) for i in range(len(df_res_rf))]
        df_res_rf['index'] = df_res_rf.index
        df_res_rf.to_csv(os.path.join(inpath, place, nam, 'df_res_rf_{0}.csv'.format(nam)), encoding='utf-16')
        """ """

        # 重要度を出力し、上位100個の変数を取り出す
        select_columns = output_importances(nam, 'rf2', x_test, clf, 100)

        # 重要度が上位の項目に絞って再学習する
        x_train, x_test = x_train[select_columns[:100]], x_test[select_columns[:100]]
        print('shape of train and test', x_train.shape, y_train.shape, x_test.shape, y_test.shape)

        dummy = DummyClassifier(strategy='most_frequent')
        dummy.fit(x_train, y_train)
        clf = RandomForestClassifier(random_state=1, n_estimators=300, min_samples_leaf=min_samples_leaf,
                                     min_impurity_decrease=min_impurity_decrease, class_weight=class_weight)
        clf = clf.fit(x_train, np.ravel(y_train))
        joblib.dump(clf, os.path.join(inpath, place, nam, 'rf3_{0}.sav'.format(nam)))
        # clf = joblib.load(os.path.join(inpath, place, nam, 'rf3_{0}.sav'.format(nam)))
        print(datetime.today(), 'fit and save last model')

        # 精度を評価する
        evaluate_all_rf.append(evaluate_model(nam, clf, 'rf3', dummy, x_train, y_train, x_test, y_test))

        # 重要度を出力する
        select_columns = output_importances(nam, 'rf3', x_test, clf, 50)
        importances_all.append(select_columns)

        # 予測値を保存する
        if i == 0:
            df_res_rf_all = df_res_rf
        else:
            df_res_rf_all = pd.concat([df_res_rf_all, df_res_rf], axis=0, sort=False)
        #
        # """ 決定木 """
        # print('\n決定木')
        # classweight = [None, 'balanced']
        # # classweight = [None, 'balanced'] if nam not in ['01', '03', '04', '08', '10', '11', '14', '15', '16', '17'] else ['balanced']
        # classweight = [None, 'balanced'] if nam not in ['14', '16', '21'] else ['balanced']
        # classweight = ['balanced'] if nam == '01' and place == '6_Phase2+alpha_maxdepth{0}'.format(depth) and depth >= 7 else classweight
        # # min_impurity_decrease 現ノードと分割後のノードの不純度の差がこの値を超える場合は分割される
        # params = {'random_state': [1],
        #           'min_samples_leaf': [25],  # 学習データの3/4のため
        #           'min_impurity_decrease': [0, 0.1, 0.01, 0.001, 0.0001, 0.00001],
        #           'max_depth': [depth],
        #           'class_weight': classweight
        #           # 'class_weight': ['balanced']
        #           }
        #
        # dummy = DummyClassifier(strategy='most_frequent')
        # dummy.fit(x_train, y_train)
        # clf = GridSearchCV(DecisionTreeClassifier(), params, cv=4, n_jobs=1, scoring='roc_auc')
        # clf.fit(x_train, np.ravel(y_train))
        # joblib.dump(clf, os.path.join(inpath, place, nam, 'dt1_{0}.sav'.format(nam)))
        # # clf = joblib.load(os.path.join(inpath, place, nam, 'dt1_{0}.sav'.format(nam)))
        # # グリッドサーチの結果を保存する
        # df_result = pd.DataFrame(clf.cv_results_)
        # df_result. \
        #     to_csv(os.path.join(inpath, place, nam, 'gridsearch_result_{0}_dt1.csv'.format(nam)))
        #
        # # 精度を評価する
        # evaluate_model(nam, clf, 'dt1', dummy, x_train, y_train, x_test, y_test)
        #
        # # AUCが最良のパラメータで再学習する
        # print('finish grid search', clf.best_params_)
        # min_impurity_decrease = clf.best_params_['min_impurity_decrease']
        # class_weight = clf.best_params_['class_weight']
        # clf = DecisionTreeClassifier(random_state=1, min_samples_leaf=33, max_depth=depth,
        #                              min_impurity_decrease=min_impurity_decrease, class_weight=class_weight)
        # clf.fit(x_train, np.ravel(y_train))
        # joblib.dump(clf, os.path.join(inpath, place, nam, 'dt2_{0}.sav'.format(nam)))
        # # clf = joblib.load(os.path.join(inpath, place, nam, 'dt2_{0}.sav'.format(nam)))
        #
        # print(datetime.today(), 'fit and save last model')
        #
        # # 精度を評価する
        # evaluate_all_dt.append(evaluate_model(nam, clf, 'dt2', dummy, x_train, y_train, x_test, y_test))
        #
        # # 可視化してPDF出力
        # output_tree(nam, clf, x_train, class_names)
        #
        # # 条件分岐をアウトプットする
        # df_result, df_res_dt = output_branch(nam, clf, class_names, x_train, y_train, x_test, y_test)
        #
        # # 条件分岐と予測値や条件Noを保存する
        # if i == 0:
        #     df_result_all = df_result
        #     df_res_dt_all = df_res_dt
        # else:
        #     df_result_all = pd.concat([df_result_all, df_result], axis=0, sort=False)
        #     df_res_dt_all = pd.concat([df_res_dt_all, df_res_dt], axis=0, sort=False)

    # 重要項目をまとめて出力する
    df_importances_all = pd.DataFrame(importances_all, index=csv_name,
                                      columns=['項目{0}'.format(i+1) for i in range(50)] * 2)
    df_importances_all.to_csv(os.path.join(inpath, place, 'importances_all.csv'),
                              encoding='utf-16')
    # # 条件分岐をまとめて出力する
    # df_result_all.to_csv(os.path.join(inpath, place, 'result_dt_all.csv'),
    #                      encoding='utf-16')

    # 評価結果をまとめて出力する
    pd.DataFrame(evaluate_all_rf).to_csv(os.path.join(inpath, place, 'evaluate_all_rf.csv'), encoding='utf-16')
    # pd.DataFrame(evaluate_all_dt).to_csv(os.path.join(inpath, place, 'evaluate_all_dt.csv'), encoding='utf-16')

    # すべての傷病組み合わせの予測値や条件Noをまとめて出力する
    df_xyorg = pd.read_csv(os.path.join(inpath, place, 'xy_org.csv'), index_col=0)
    df_xyorg['index'] = df_xyorg.index

    df_res_rf_all = pd.merge(df_xyorg[['index', 'a_id', 'b_id', 'a_tol_flg']], df_res_rf_all)
    df_res_rf_all.to_csv(os.path.join(inpath, place, 'df_res_rf_all.csv'), index=None, encoding='utf-16')

    # df_res_dt_all = pd.merge(df_xyorg[['index', 'a_id', 'b_id', 'a_tol_flg']], df_res_dt_all)
    # df_res_dt_all.to_csv(os.path.join(inpath, place, 'df_res_dt_all.csv'), index=None, encoding='utf-16')


def evaluate_model(nam, model, model_name, dummy, x_train, y_train, x_test=None, y_test=None):

    pred_train = model.predict(x_train)
    proba_train = model.predict_proba(x_train)
    proba_dummy = dummy.predict_proba(x_train)[0]
    if x_test is not None:
        pred_test = model.predict(x_test)
        proba_test = model.predict_proba(x_test)

    print(datetime.today(),
          'confusion matrix  upper left:both notol  upper right:real notol/pred tol  '
          'lower left:real tol/pred notol  lower right:both tol')
    con_mat_tr = confusion_matrix(y_train, pred_train)
    con_mat_ts = confusion_matrix(y_test, pred_test) if x_test is not None else ''
    cla_rep_tr = classification_report(y_train, pred_train, target_names=['notol', 'tol'], digits=5)
    cla_rep_ts = classification_report(y_test, pred_test, target_names=['notol', 'tol'], digits=5)
    print(con_mat_tr)
    print(con_mat_ts)
    print(cla_rep_tr)
    print(cla_rep_ts)

    # 学習データとダミーデータのAUC ROC
    fpr_tr, tpr_tr, thresholds_tr = roc_curve(y_train, proba_train[:, 1])
    fpr_dm, tpr_dm, thresholds_dm = roc_curve(y_train, proba_dummy[:, 1])
    roc_auc_tr = auc(fpr_tr, tpr_tr)
    auc_train = roc_auc_score(y_train, proba_train[:, 1:])  # roc_auc_trとauc_trainは同じ
    print('AUC_train', round(roc_auc_tr, 4))

    # 評価データのAUC ROC
    if x_test is not None:
        fpr_ts, tpr_ts, thresholds_ts = roc_curve(y_test, proba_test[:, 1])
        roc_auc_ts = auc(fpr_ts, tpr_ts)
        auc_test = roc_auc_score(y_test, proba_test[:, 1:])  # roc_auc_tsとauc_testは同じ
        print('AUC_test', round(roc_auc_ts, 4))

    # テキストに保存する
    with open(os.path.join(inpath, place, nam, 'evaluate_{0}_{1}.txt'.format(nam, model_name)), mode='w') as f:
        f.write('train_shape' + str(x_train.shape) + '\n')
        f.write('test_shape' + str(x_test.shape) + '\n')
        f.write(str(model.get_params(deep=False)) + '\n')
        f.write('Confusion Matrix' + '\n')
        f.write(str(con_mat_tr[0][0]) + ',' + str(con_mat_tr[0][1]) + '\n')
        f.write(str(con_mat_tr[1][0]) + ',' + str(con_mat_tr[1][1]) + '\n')
        f.write(str(con_mat_ts[0][0]) + ',' + str(con_mat_ts[0][1]) + '\n')
        f.write(str(con_mat_ts[1][0]) + ',' + str(con_mat_ts[1][1]) + '\n')
        f.write(str(cla_rep_tr) + '\n')
        f.write(str(cla_rep_ts) + '\n')
        f.write('AUC_train' + ',' + str(round(roc_auc_tr, 4)) + '\n')
        f.write('AUC_test' + ',' + str(round(roc_auc_ts, 4)))

    # ROC曲線
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(fpr_tr, tpr_tr, color='b', label='train (area={0})'.format(round(roc_auc_tr, 3)))
    if x_test is not None:
        ax1.plot(fpr_ts, tpr_ts, color='r', label='test (area={0})'.format(round(roc_auc_ts, 3)))
    ax1.plot(fpr_dm, tpr_dm, color='orange', label='dummy')
    ax1.set_xlabel('False Positivie Rate: 1-Recall of nototal')
    ax1.set_ylabel('True Positivie Rate: Recall of total')
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlim([0.0, 1.0])
    ax1.set_title('Receiver operating characteristic example')
    ax1.legend(loc='lower right')

    # 保存する
    plt.savefig(os.path.join(inpath, place, nam, 'ROCcurve_{0}_{1}.png'.format(nam, model_name)))

    result = [con_mat_ts[1, 1], con_mat_ts[1, 0], con_mat_ts[0, 1], con_mat_ts[0, 0],
              round(roc_auc_tr, 4), round(roc_auc_ts, 4)]
    return result


def output_importances(nam, model, x, clf, select_num=None):
    # 特徴量の重要度を取得する
    feature_imps = clf.feature_importances_
    print(len(feature_imps), x.shape)
    # 特徴量の名前
    label = x.columns[0:]
    label_jp = convert_jp.convert_jp(label, 0)
    # 必要な項目抽出用
    select_columns = []
    # 特徴量の重要度順（降順）
    indices = np.argsort(feature_imps)[::-1]

    with open(os.path.join(inpath, place, nam, 'importances_{0}_{1}.csv'.format(nam, model)), mode='w') as f:
        for i in range(0, select_num):

            # 上位100の変数を出力する＆取り出す
            line = str(i + 1) + "," + str(label[indices[i]]) + "," + str(feature_imps[indices[i]]) \
                   + "," + str(label_jp[indices[i]])
            # print(line)
            f.write(str(line) + '\n')

            select_columns.append(str(label[indices[i]]))

    for i in range(0, select_num):
        select_columns.append(str(label_jp[indices[i]]))

    return select_columns


def output_tree(nam, dt, x, class_names):
    dot_data = tree.export_graphviz(dt,  # 決定木オブジェクトを一つ指定する
                                    out_file=None,  # ファイルは介さずにGraphvizにdot言語データを渡すのでNone
                                    filled=True,  # Trueにすると、分岐の際にどちらのノードに多く分類されたのか色で示してくれる
                                    rounded=True,  # Trueにすると、ノードの角を丸く描画する。
                                    feature_names=x.columns,  # これを指定しないとチャート上で特徴量の名前が表示されない
                                    class_names=class_names,  # これを指定しないとチャート上で分類名が表示されない
                                    special_characters=True  # 特殊文字を扱えるようにする
                                    )
    graph = pydotplus.graphviz.graph_from_dot_data(dot_data)
    graph.write_png(os.path.join(inpath, place, nam, 'tree_{0}.png'.format(nam)))


def get_result_dt(dt):
    left = dt.tree_.children_left
    right = dt.tree_.children_right
    get_left = []
    max_depth = depth + 1  # 出力する深さ
    # result_row = np.full(int(math.log2(max(right))) + 1, -1)
    # result_row_yn = np.full(int(math.log2(max(right))) + 1, '   ')
    result_row = np.full(max_depth, -1)
    result_row_yn = np.full(max_depth, '   ')
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
    result = result.reshape(-1, max_depth)
    result_yn = result_yn.reshape(-1, max_depth)

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
    print(datetime.today(), 'select all 0 columns Done. start delete columns')
    df = df.drop(drop_columns, axis=1)

    return df


def output_branch(nam, dt, class_names, x_train, y_train, x_test=None, y_test=None):
    # 分岐条件の変数名（IDと名称の対応は学習データの順番？）
    # -2は分岐条件がない（リーフである）ことを示す
    branch_name = [x_train.columns[i] if i != -2 else '' for i in dt.tree_.feature]
    branch_name_jp = convert_jp.convert_jp(branch_name, 1, dt.tree_.threshold)

    # 到達ノードと予測クラスを取得する
    # # 学習データ
    path_train = dt.apply(x_train)
    pred_train = dt.predict(x_train)
    # proba_train = dt.predict_proba(x_train)
    # # 評価データ
    if x_test is not None:
        path_test = dt.apply(x_test)
        pred_test = dt.predict(x_test)
        proba_test = dt.predict_proba(x_test)

    # 分岐条件を二次元で取得する
    result, result_yn, last_node = get_result_dt(dt)

    result_jp = []
    result_last_node = []
    all_impurity = 1 - ((y_train['y'] == 0).sum() / len(np.ravel(y_train))) ** 2\
                     - ((y_train['y'] == 1).sum() / len(np.ravel(y_train))) ** 2

    """ """
    pred_path_proba_test = np.c_[pred_test, proba_test[:, 1], proba_test[:, 0], path_test]
    df_res_dt = pd.DataFrame(pred_path_proba_test, index=y_test.index,
                             columns=['pred', 'proba_tol', 'proba_notol', 'path'])
    df_res_dt['y'] = y_test['y']
    df_res_dt['index'] = df_res_dt.index
    df_res_dt['groupno'] = [int(nam) for i in range(len(df_res_dt))]
    paths = df_res_dt['path'].drop_duplicates().astype(int).sort_values().reset_index(drop=True)
    paths_jp = pd.Series(['条件{0}'.format(i+1) for i in range(len(paths))])
    df_paths = pd.DataFrame(pd.concat([paths, paths_jp], axis=1))
    df_paths.columns = ['path', 'path_jp']
    df_res_dt = pd.merge(df_res_dt, df_paths)

    df_res_dt.to_csv(os.path.join(inpath, place, nam, 'df_res_dt_{0}.csv'.format(nam)), encoding='utf-16')
    """ """

    for i in range(result.shape[0]):
        row_branch, row_threshold, row_impurity, row_samples, row_value, row_class, row_last_node = \
            [], [], [], [], [], [], []

        for j, k in zip(result[i, :].astype(int), range(result_yn.shape[1])):
            if dt.tree_.threshold[j] != -2.0:
                row_branch.append('{0} {1}'.format(branch_name_jp[j], result_yn[i, k]))
                row_threshold.append(dt.tree_.threshold[j])
                # row_impurity.append(round(dt.tree_.impurity[j], 3))
                # row_samples.append(dt.tree_.n_node_samples[j])
                # row_value.append(dt.tree_.value[j][0])
                # row_class.append(class_names[np.argmax(dt.tree_.value[j])])
            else:
                row_branch.append('')
                row_threshold.append('')
                # row_impurity.append('')
                # row_samples.append('')
                # row_value.append('')
                # row_class.append('')

        # result_jp.append(row_branch + row_threshold + row_impurity + row_samples + row_value + row_class)
        result_jp.append(row_branch + row_threshold)

        # 予測クラス
        row_last_node.append(class_names[np.argmax(dt.tree_.value[last_node[i]])])
        # 全体の不純度
        row_last_node.append(all_impurity)
        # 学習データ
        # # 最終ノードの到達数
        row_last_node.append((path_train == last_node[i]).astype(int).sum())
        # # 最終ノードのクラス別到達数
        row_last_node.append(((path_train == last_node[i]) & (pred_train == 0)).astype(int).sum())
        row_last_node.append(((path_train == last_node[i]) & (pred_train == 1)).astype(int).sum())
        # # 実際のクラス別数
        row_last_node.append(((path_train == last_node[i]) & (y_train['y'] == 0)).astype(int).sum())
        row_last_node.append(((path_train == last_node[i]) & (y_train['y'] == 1)).astype(int).sum())
        # # 不純度
        row_last_node.append(1 - (((path_train == last_node[i]) & (y_train['y'] == 0)).astype(int).sum() /
                                  (path_train == last_node[i]).astype(int).sum()) ** 2
                               - (((path_train == last_node[i]) & (y_train['y'] == 1)).astype(int).sum() /
                                  (path_train == last_node[i]).astype(int).sum()) ** 2)

        # 評価データ
        if x_test is not None:
            # # 最終ノードの到達数
            row_last_node.append((path_test == last_node[i]).astype(int).sum())
            # # 最終ノードのクラス別到達数
            row_last_node.append(((path_test == last_node[i]) & (pred_test == 0)).astype(int).sum())
            row_last_node.append(((path_test == last_node[i]) & (pred_test == 1)).astype(int).sum())
            # # 実際のクラス別数
            row_last_node.append(((path_test == last_node[i]) & (y_test['y'] == 0)).astype(int).sum())
            row_last_node.append(((path_test == last_node[i]) & (y_test['y'] == 1)).astype(int).sum())
            # # 不純度
            row_last_node.append(1 - (((path_test == last_node[i]) & (y_test['y'] == 0)).astype(int).sum() /
                                      (path_test == last_node[i]).astype(int).sum()) ** 2
                                   - (((path_test == last_node[i]) & (y_test['y'] == 1)).astype(int).sum() /
                                      (path_test == last_node[i]).astype(int).sum()) ** 2)

        result_last_node.append(row_last_node)

    df_result_jp = pd.DataFrame(result_jp)
    df_result_last_node = pd.DataFrame(result_last_node)

    df_result = pd.concat([df_result_jp, df_result_last_node], axis=1)
    df_result.index = ['条件{0}'.format(i) for i in range(1, df_result_jp.shape[0] + 1)]

    columns_branch = ['項目{0}_条件'.format(i) for i in range(1, int(df_result_jp.shape[1] / 2) + 1)]
    columns_threshold = ['項目{0}_閾値'.format(i) for i in range(1, int(df_result_jp.shape[1] / 2) + 1)]
    # columns_impurity = ['項目{0}_gini'.format(i) for i in range(1, int(df_result_jp.shape[1] / 6) + 1)]
    # columns_samples = ['項目{0}_到達数'.format(i) for i in range(1, int(df_result_jp.shape[1] / 6) + 1)]
    # columns_value = ['項目{0}_クラス別到達数'.format(i) for i in range(1, int(df_result_jp.shape[1] / 6) + 1)]
    # columns_class = ['項目{0}_予測クラス'.format(i) for i in range(1, int(df_result_jp.shape[1] / 6) + 1)]
    columns2 = ['pred_class', 'all_impurity', 'train_samples', 'train_pred_non_tol', 'train_pred_tol',
                'train_real_non_tol', 'train_real_tol', 'train_impurity',
                # 'train_AUC',
                'test_samples', 'test_pred_non_tol', 'test_pred_tol',
                # 'test_AUC',
                'test_real_non_tol', 'test_real_tol', 'test_impurity'] if x_test is not None else \
               ['pred_class', 'train_samples', 'train_pred_non_tol', 'train_pred_tol',
                'train_real_non_tol', 'train_real_tol', 'train_impurity'
                # ,'train_AUC'
                ]
    # df_result.columns = columns_branch + columns_threshold + columns_impurity + columns_samples + columns_value + \
    #                     columns_class + columns2
    df_result.columns = columns_branch + columns_threshold + columns2

    # 適合率のカラムを作成する
    # # 学習データ
    df_tmp = df_result[['pred_class', 'train_samples', 'train_real_non_tol', 'train_real_tol']]
    df_tmp.columns = ['pred_class', 'samples', 'real_non_tol', 'real_tol']
    df_result['train_precision'] = df_tmp.apply(lambda x: calc_precision(x), axis=1)
    # # 評価データ
    if x_test is not None:
        df_tmp = df_result[['pred_class', 'test_samples', 'test_real_non_tol', 'test_real_tol']]
        df_tmp.columns = ['pred_class', 'samples', 'real_non_tol', 'real_tol']
        df_result['test_precision'] = df_tmp.apply(lambda x: calc_precision(x), axis=1)

    df_result.to_csv(os.path.join(inpath, place, nam, 'result_dt_{0}.csv'.format(nam)),
                     encoding='utf-16')

    df_result['groupno'] = [nam for i in range(df_result.shape[0])]

    return df_result, df_res_dt


def calc_precision(x):
    if x.pred_class == 'non_tol':
        return x.real_non_tol / x.samples
    else:
        return x.real_tol / x.samples


def drop_date_from_b_nyutai(x):
    # 過去請求の入院日または退院日から 今回診断書の日付項目までの日数を計算する
    x = x.drop(columns=['byo_date_from_bnyu_max']) if 'byo_date_from_bnyu_max' in x.columns else x
    x = x.drop(columns=['byo_date_from_bnyu_min']) if 'byo_date_from_bnyu_min' in x.columns else x
    x = x.drop(columns=['byo_date_from_btai_max']) if 'byo_date_from_btai_max' in x.columns else x
    x = x.drop(columns=['byo_date_from_btai_min']) if 'byo_date_from_btai_min' in x.columns else x
    x = x.drop(columns=['syosin_date_from_bnyu_max']) if 'syosin_date_from_bnyu_max' in x.columns else x
    x = x.drop(columns=['syosin_date_from_bnyu_min']) if 'syosin_date_from_bnyu_min' in x.columns else x
    x = x.drop(columns=['syosin_date_from_btai_max']) if 'syosin_date_from_btai_max' in x.columns else x
    x = x.drop(columns=['syosin_date_from_btai_min']) if 'syosin_date_from_btai_min' in x.columns else x
    x = x.drop(columns=['gankaku_date_from_bnyu_max']) if 'gankaku_date_from_bnyu_max' in x.columns else x
    x = x.drop(columns=['gankaku_date_from_bnyu_min']) if 'gankaku_date_from_bnyu_min' in x.columns else x
    x = x.drop(columns=['gankaku_date_from_btai_max']) if 'gankaku_date_from_btai_max' in x.columns else x
    x = x.drop(columns=['gankaku_date_from_btai_min']) if 'gankaku_date_from_btai_min' in x.columns else x
    x = x.drop(columns=['hokagankaku_date_from_bnyu_max']) if 'hokagankaku_date_from_bnyu_max' in x.columns else x
    x = x.drop(columns=['hokagankaku_date_from_bnyu_min']) if 'hokagankaku_date_from_bnyu_min' in x.columns else x
    x = x.drop(columns=['hokagankaku_date_from_btai_max']) if 'hokagankaku_date_from_btai_max' in x.columns else x
    x = x.drop(columns=['hokagankaku_date_from_btai_min']) if 'hokagankaku_date_from_btai_min' in x.columns else x

    return x


def separate_data(x, y, value, columns):
    """ モデルの作成単位を分割する """

    for num, column in columns.items():
        if isinstance(column, list):
            x_yes = x[(x[column[0]] > value) & (x[column[1]] > value)]
            x_org = x[(x[column[0]] <= value) | (x[column[1]] <= value)]
        else:
            x_yes = x[(x[column] > value) & (x[column] > value)]
            x_org = x[(x[column] <= value) | (x[column] <= value)]

        y_yes = y.loc[x_yes.index, :]
        y_org = y.loc[x_org.index, :]

        x_yes = drop_all0_columns(x_yes)
        x_org = drop_all0_columns(x_org)

        os.mkdir(os.path.join(inpath, place, num))
        y_yes.to_csv(os.path.join(inpath, place, num, 'y_{0}.csv'.format(num)), encoding='utf-8')
        x_yes.to_csv(os.path.join(inpath, place, num, 'x_{0}.csv'.format(num)), encoding='utf-8')

        x = x_org
        y = y_org
    os.mkdir(os.path.join(inpath, place, str(int(num)+1)))
    y.to_csv(os.path.join(inpath, place, str(int(num)+1), 'y_{0}.csv'.format(str(int(num)+1))), encoding='utf-8')
    x.to_csv(os.path.join(inpath, place, str(int(num)+1), 'x_{0}.csv'.format(str(int(num)+1))), encoding='utf-8')
    exit()

    return None


if __name__ == "__main__":

    print(datetime.today(), 'START')

    main()

    print(datetime.today(), 'END')


