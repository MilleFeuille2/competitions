# -*- encoding:utf-8 -*-
import csv
import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.grid_search import GridSearchCV


num = 100000


def main():
    print(datetime.today())

    """ 今回請求IDと過去請求IDごとの査定データをコード化データ付きで取得する """
    sql_get_satei = get_satei()
    df_satei = pd.read_sql(sql=sql_get_satei, con=conn, index_col=None)

    """ 今回請求IDごとの診断書データをコード化データ付きで取得する """
    sql_get_certificate_now = get_certificate_now()
    df_cer_now = pd.read_sql(sql=sql_get_certificate_now, con=conn, index_col='id_')
    """ 過去請求IDごとの請求書データをコード化データ付きで取得する """
    sql_get_certificate_bef = get_certificate_bef()
    df_cer_bef = pd.read_sql(sql=sql_get_certificate_bef, con=conn, index_col='id_')

    """ 一意な今回請求IDと過去請求IDの組を作成する """
    df_ids = df_satei[['a_id', 'b_id', 'a_tol_flg']].drop_duplicates()
    print('今回請求IDと過去請求IDの組', df_ids.shape)
    # TODO HOME drop_duplicates()は不要なはずだが念のためつける場合とつけない場合を比較する
    df_ids = df_satei[['a_id', 'b_id', 'a_tol_flg']]
    print('今回請求IDと過去請求IDの組', df_ids.shape)

    """ 傷病コードリストと手術コードリストを作成する """
    list_byo = make_list_byo(df_satei, df_cer_now, df_cer_bef)
    list_ope = make_list_ope(df_cer_now, df_cer_bef)

    # 査定データの加工処理
    df_res_satei_now = process_satei(df_satei, list_byo)
    df_res_satei_bef = process_satei(df_satei, list_byo)

    # 診断書データの加工処理
    df_cer_byo_code_now, df_cer_ope_code_now, df_cer_other_now, ids_now =\
        process_certificate(df_cer_now, list_byo, list_ope, 'a')
    df_cer_byo_code_bef, df_cer_ope_code_bef, df_cer_other_bef, ids_bef =\
        process_certificate(df_cer_bef, list_byo, list_ope, 'b')

    df_res_satei.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_satei\df_satei.csv')
    df_res_now.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_certificate\df_res_now.csv')
    df_res_bef.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_certificate\df_res_bef.csv')

    # データ加工処理結果を結合する（今回と過去いずれかでも診断書が0件の場合は除外するため、内部結合）
    df_result = pd.merge(df_res_satei[['a_id', 'b_id', 'a_tol_flg']], df_res_now, left_on='a_id', right_on='id_now')
    df_result = pd.merge(df_result, df_res_bef, left_on='b_id', right_on='id_bef')

    df_result.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_certificate\df_result.csv')

    x = df_result.iloc[:, 5:].drop('id_bef', axis=1)
    y = df_result['a_tol_flg']

    # # TODO モデルは別PGのため後で削除する
    # print(df_result.iloc[:, 2:].shape, df_ids['a_tol_flg'].shape)
    #
    # x = df_result.iloc[:, 2:]
    # y = pd.Series(df_ids['a_tol_flg'])

    # df_result.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_certificate\df_result.csv')
    # x.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_certificate\x.csv')
    # y.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_certificate\y.csv')
    # df_ids.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_certificate\df_ids.csv')

    x_train, x_test = train_test_split(x, random_state=1)
    y_train, y_test = train_test_split(y, random_state=1)

    print(datetime.today())

    # clf = RandomForestClassifier(random_state=0)
    # clf = RandomForestClassifier(random_state=0, n_estimators=100)
    clf = RandomForestClassifier(random_state=0, n_estimators=300)
    clf.fit(x_train, y_train)

    score_train = clf.score(x_train, y_train)
    score_test = clf.score(x_test, y_test)
    print(score_train, score_test)

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

    # # 特徴量の重要度を取得する
    # feature_imps = clf.feature_importances_
    # print(len(feature_imps), x.shape)
    # # 特徴量の名前
    # label = x.columns[0:]
    # # 特徴量の重要度順（降順）
    # indices = np.argsort(feature_imps)[::-1]
    # for i in range(len(feature_imps)):
    #     print(str(i + 1) + "   " + str(label[indices[i]]) + "   " + str(feature_imps[indices[i]]))
    #
    # # confusion matrix
    # pred = clf.predict(x_test)
    # print('予測結果', pred[:3])
    # print('実測結果', y_test[:3])
    # print(confusion_matrix(y_test, clf.predict(x_test)))
    # print(accuracy_score(y_test, clf.predict(x_test)))


def get_satei():
    # 査定データとコード化データを、今回請求と過去請求それぞれ結合して取得する
    return 'SELECT ' \
           ' tol*, ' \
           ' main1_now.totalgroup code_main1_now, ' \
           ' main2_now.totalgroup code_main2_now, ' \
           ' cause1_now.totalgroup code_cause1_now, ' \
           ' cause2_now.totalgroup code_cause2_now, ' \
           ' gappei_now.totalgroup code_gappei_now, ' \
           ' kiou_now.totalgroup code_kiou_now, ' \
           ' main1_bef.totalgroup code_main1_bef, ' \
           ' main2_bef.totalgroup code_main2_bef, ' \
           ' cause1_bef.totalgroup code_cause1_bef, ' \
           ' cause2_bef.totalgroup code_cause2_bef, ' \
           ' gappei_bef.totalgroup code_gappei_bef, ' \
           ' kiou_bef.totalgroup code_kiou_bef ' \
           'FROM daido_total_opportunity as tol' \
           'LEFT JOIN code_main1 as main1_now'\
           'on main1_now.id_ = tol.id_ ' \
           'LEFT JOIN code_main2 as main2_now'\
           'on main2_now.id_ = tol.id_ ' \
           'LEFT JOIN code_cause1 as cause1_now'\
           'on cause1_now.id_ = tol.id_ ' \
           'LEFT JOIN code_cause2 as cause2_now'\
           'on cause2_now.id_ = tol.id_ ' \
           'LEFT JOIN code_gappei as gappei_now'\
           'on gappei_now.id_ = tol.id_ ' \
           'LEFT JOIN code_kiou as kiou_now'\
           'on kiou_now.id_ = tol.id_ ' \
           'LEFT JOIN code_main1 as main1_bef'\
           'on main1_bef.id_ = tol.id_ ' \
           'LEFT JOIN code_main2 as main2_bef'\
           'on main2_bef.id_ = tol.id_ ' \
           'LEFT JOIN code_cause1 as cause1_bef'\
           'on cause1_bef.id_ = tol.id_ ' \
           'LEFT JOIN code_cause2 as cause2_bef'\
           'on cause2_bef.id_ = tol.id_ ' \
           'LEFT JOIN code_gappei as gappei_bef'\
           'on gappei_bef.id_ = tol.id_ ' \
           'LEFT JOIN code_kiou as kiou_bef'\
           'on kiou_bef.id_ = tol.id_ ' \
           'ORDER BY tol.id_ '


def get_certificate_now():
    # 今回請求IDに紐づく診断書データをコード付きで取得する
    return 'select ' \
           ' cer.*, ' \
           ' byo1.totalgroup code_byo1,' \
           ' byo2.totalgroup code_byo2,' \
           ' byo3.totalgroup code_byo3,' \
           ' byo4.totalgroup code_byo4,' \
           ' byo5.totalgroup code_byo5,' \
           ' byogen1.totalgroup code_byogen1,' \
           ' byogen2.totalgroup code_byogen2,' \
           ' byogen3.totalgroup code_byogen3,' \
           ' gappei1.totalgroup code_gappei1,' \
           ' gappei2.totalgroup code_gappei2,' \
           ' gappei3.totalgroup code_gappei3,' \
           ' hosya.totalgroup code_hosya,' \
           ' byori.totalgroup code_byori,' \
           ' hokabyori.totalgroup code_hokabyori,' \
           ' gappei_ope.totalgroup code_gappei_ope,' \
           ' ope1.totalgroup code_ope1,' \
           ' ope2.totalgroup code_ope2,' \
           ' ope3.totalgroup code_ope3,' \
           ' ope4.totalgroup code_ope4,' \
           ' ope5.totalgroup code_ope5' \
           ' from  ' \
           ' daido_certificate_now as cer' \
           ' LEFT JOIN code_byomei1 as byo1' \
           ' on byo1.id_no = cer.id_no' \
           ' LEFT JOIN code_byomei2 as byo2' \
           ' on byo2.id_no = cer.id_no' \
           ' LEFT JOIN code_byomei3 as byo3' \
           ' on byo3.id_no = cer.id_no' \
           ' LEFT JOIN code_byomei4 as byo4' \
           ' on byo4.id_no = cer.id_no' \
           ' LEFT JOIN code_byomei5 as byo5' \
           ' on byo5.id_no = cer.id_no' \
           ' LEFT JOIN code_byogenin1 as byogen1' \
           ' on byogen1.id_no = cer.id_no' \
           ' LEFT JOIN code_byogenin2 as byogen2' \
           ' on byogen2.id_no = cer.id_no' \
           ' LEFT JOIN code_byogenin3 as byogen3' \
           ' on byogen3.id_no = cer.id_no' \
           ' LEFT JOIN code_gappei1 as gappei1' \
           ' on gappei1.id_no = cer.id_no' \
           ' LEFT JOIN code_gappei2 as gappei2' \
           ' on gappei2.id_no = cer.id_no' \
           ' LEFT JOIN code_gappei3 as gappei3' \
           ' on gappei3.id_no = cer.id_no' \
           ' LEFT JOIN code_hosyabui as hosya' \
           ' on hosya.id_no = cer.id_no' \
           ' LEFT JOIN code_byorisin as byori' \
           ' on byori.id_no = cer.id_no' \
           ' LEFT JOIN code_hokabyorisin as hokabyori' \
           ' on hokabyori.id_no = cer.id_no' \
           ' LEFT JOIN code_gappei_ope as gappei_ope' \
           ' on gappei_ope.id_no = cer.id_no' \
           ' LEFT JOIN code_ope1 as ope1' \
           ' on ope1.id_no = cer.id_no' \
           ' LEFT JOIN code_ope2 as ope2' \
           ' on ope2.id_no = cer.id_no' \
           ' LEFT JOIN code_ope3 as ope3' \
           ' on ope3.id_no = cer.id_no' \
           ' LEFT JOIN code_ope4 as ope4' \
           ' on ope4.id_no = cer.id_no' \
           ' LEFT JOIN code_ope5 as ope5' \
           ' on ope5.id_no = cer.id_no' \
           ' order by cer.id_no' \
           ' ' \
           ' offset 0 limit {0} '.format(num)


def get_certificate_bef():
    # 過去請求IDに紐づく診断書データをコード付きで取得する
    return 'select ' \
           ' cer.*, ' \
           ' byo1.totalgroup code_byo1,' \
           ' byo2.totalgroup code_byo2,' \
           ' byo3.totalgroup code_byo3,' \
           ' byo4.totalgroup code_byo4,' \
           ' byo5.totalgroup code_byo5,' \
           ' byogen1.totalgroup code_byogen1,' \
           ' byogen2.totalgroup code_byogen2,' \
           ' byogen3.totalgroup code_byogen3,' \
           ' gappei1.totalgroup code_gappei1,' \
           ' gappei2.totalgroup code_gappei2,' \
           ' gappei3.totalgroup code_gappei3,' \
           ' hosya.totalgroup code_hosya,' \
           ' byori.totalgroup code_byori,' \
           ' hokabyori.totalgroup code_hokabyori,' \
           ' gappei_ope.totalgroup code_gappei_ope,' \
           ' ope1.totalgroup code_ope1,' \
           ' ope2.totalgroup code_ope2,' \
           ' ope3.totalgroup code_ope3,' \
           ' ope4.totalgroup code_ope4,' \
           ' ope5.totalgroup code_ope5' \
           ' from  ' \
           ' daido_certificate_bef as cer' \
           ' LEFT JOIN code_byomei1 as byo1' \
           ' on byo1.id_no = cer.id_no' \
           ' LEFT JOIN code_byomei2 as byo2' \
           ' on byo2.id_no = cer.id_no' \
           ' LEFT JOIN code_byomei3 as byo3' \
           ' on byo3.id_no = cer.id_no' \
           ' LEFT JOIN code_byomei4 as byo4' \
           ' on byo4.id_no = cer.id_no' \
           ' LEFT JOIN code_byomei5 as byo5' \
           ' on byo5.id_no = cer.id_no' \
           ' LEFT JOIN code_byogenin1 as byogen1' \
           ' on byogen1.id_no = cer.id_no' \
           ' LEFT JOIN code_byogenin2 as byogen2' \
           ' on byogen2.id_no = cer.id_no' \
           ' LEFT JOIN code_byogenin3 as byogen3' \
           ' on byogen3.id_no = cer.id_no' \
           ' LEFT JOIN code_gappei1 as gappei1' \
           ' on gappei1.id_no = cer.id_no' \
           ' LEFT JOIN code_gappei2 as gappei2' \
           ' on gappei2.id_no = cer.id_no' \
           ' LEFT JOIN code_gappei3 as gappei3' \
           ' on gappei3.id_no = cer.id_no' \
           ' LEFT JOIN code_hosyabui as hosya' \
           ' on hosya.id_no = cer.id_no' \
           ' LEFT JOIN code_byorisin as byori' \
           ' on byori.id_no = cer.id_no' \
           ' LEFT JOIN code_hokabyorisin as hokabyori' \
           ' on hokabyori.id_no = cer.id_no' \
           ' LEFT JOIN code_gappei_ope as gappei_ope' \
           ' on gappei_ope.id_no = cer.id_no' \
           ' LEFT JOIN code_ope1 as ope1' \
           ' on ope1.id_no = cer.id_no' \
           ' LEFT JOIN code_ope2 as ope2' \
           ' on ope2.id_no = cer.id_no' \
           ' LEFT JOIN code_ope3 as ope3' \
           ' on ope3.id_no = cer.id_no' \
           ' LEFT JOIN code_ope4 as ope4' \
           ' on ope4.id_no = cer.id_no' \
           ' LEFT JOIN code_ope5 as ope5' \
           ' on ope5.id_no = cer.id_no' \
           ' order by cer.id_no' \
           ' ' \
           ' offset 0 limit {0} '.format(num)


def make_list_byo(df_satei, df_cer_now, df_cer_bef):
    # 傷病コードのリストを作成する（査定データ、診断書データ共通）
    list_byo = \
        pd.concat((df_satei['code_main1'], df_satei['code_main2'], df_satei['code_cause1'], df_satei['code_cause2'],
                   df_satei['code_gappei'], df_satei['code_kiou'],
                   df_cer_now['code_byo1'], df_cer_now['code_byo2'], df_cer_now['code_byo3'],
                   df_cer_now['code_byo4'], df_cer_now['code_byo5'],
                   df_cer_now['code_byogen1'], df_cer_now['code_byogen2'], df_cer_now['code_byogen3'],
                   df_cer_now['code_gappei1'], df_cer_now['code_gappei2'], df_cer_now['code_gappei3'],
                   df_cer_now['hosya'], df_cer_now['byori'], df_cer_now['hokabyori'],
                   df_cer_bef['code_byo1'], df_cer_bef['code_byo2'], df_cer_bef['code_byo3'],
                   df_cer_bef['code_byo4'], df_cer_bef['code_byo5'],
                   df_cer_bef['code_byogen1'], df_cer_bef['code_byogen2'], df_cer_bef['code_byogen3'],
                   df_cer_bef['code_gappei1'], df_cer_bef['code_gappei2'], df_cer_bef['code_gappei3'],
                   df_cer_bef['hosya'], df_cer_bef['byori'], df_cer_bef['hokabyori']),
                  axis=0)
    list_byo = list_byo.drop_duplicates().sort_values()
    return list_byo[2:]  # ''と'000'を除外する


def make_list_ope(df_cer_now, df_cer_bef):
    # 手術コードのリストを作成する
    list_ope = \
        pd.concat((df_cer_now['code_gappei_ope'],
                   df_cer_now['code_ope1'], df_cer_now['code_ope2'], df_cer_now['code_ope3'],
                   df_cer_now['code_ope4'], df_cer_now['code_ope5'],
                   df_cer_bef['code_ope1'], df_cer_bef['code_ope2'], df_cer_bef['code_ope3'],
                   df_cer_bef['code_ope4'], df_cer_bef['code_ope5']),
                  axis=0)
    list_ope = list_ope.drop_duplicates().sort_values()
    return list_ope[2:]  # ''と'000'を除外する


def process_satei(df, list_byo):

    df_res = df[['a_id', 'b_id']]

    # 今回請求と過去請求のコードを比較して変数を作成する
    # 今回主傷病と過去主傷病
    # 今回主傷病と過去原因傷病
    # 今回主傷病と過去合併症
    # 今回主傷病と過去既往症
    # 今回原因傷病と過去主傷病
    # 今回原因傷病と過去原因傷病
    # 今回原因傷病と過去合併症
    # 今回原因傷病と過去既往症
    # 今回合併症と過去主傷病
    # 今回合併症と過去原因傷病
    # 今回合併症と過去合併症
    # 今回合併症と過去既往症
    # 今回既往症と過去主傷病
    # 今回既往症と過去原因傷病
    # 今回既往症と過去合併症
    # 今回既往症と過去既往症

    # 今回請求のダミー変数を作成する
    df_dummy_now = process_satei_dummy(df, list_byo, 'now')

    # 過去請求のダミー変数を作成する
    df_dummy_bef = process_satei_dummy(df, list_byo, 'bef')
    
    # 主傷病を中分類化してダミー変数化
    # 主傷病の中分類が同じかフラグ
    # がん区分や生活習慣病区分が同じかフラグ

    # 結合する

    return df_res


def process_satei_dummy(df, list_byo, when):
    df_res = df[['a_id', 'b_id']]
    df_tmp = df[['a_id', 'b_id']]

    for code in list_byo:
        # 査定データのダミー変数化
        df_tmp['code_main1_{0}'.format(code)] = df['code_main1_{0}'.format(when)] == code
        df_tmp['code_main2_{0}'.format(code)] = df['code_main2_{0}'.format(when)] == code
        df_res['code_main_{0}'.format(code)] = int(df_tmp['code_main1_{0}'.format(code)] +
                                                   df_tmp['code_main2_{0}'.format(code)])
        df_tmp['code_cause1_{0}'.format(code)] = df['code_cause1_{0}'.format(when)] == code
        df_tmp['code_cause2_{0}'.format(code)] = df['code_cause2_{0}'.format(when)] == code
        df_res['code_cause_{0}'.format(code)] = int(df_tmp['code_cause1_{0}'.format(code)] +
                                                    df_tmp['code_cause2_{0}'.format(code)])
        df_res['code_gappei_{0}'.format(code)] = int(df['code_gappei_{0}'.format(when)] == code)
        df_res['code_kiou_{0}'.format(code)] = int(df['code_kiou_{0}'.format(when)] == code)

    df_res = drop_all0_columns(df_res)
    return df_res


def process_certificate(df, list_byo, list_ope, when):
    
    # 日付で判断して無効な値は'000'などにする（手術名など）
    # できれば列単位で処理したい
    
    # id_no単位でダミー変数化する
    df_dummy_byo, df_dummy_ope = process_certificate_dummy(df, list_byo, list_ope, when)

    # ダミー変数化したものをid単位にする
    df_dummy_byo = process_certificate_dummy_groupid(df_dummy_byo)
    df_dummy_ope = process_certificate_dummy_groupid(df_dummy_ope)

    """ 傷病・手術コードのダミー変数以外の変数を作成する """
    # 傷病発生日との期間
    # 入院必要性有無
    # 前医有無
    # 既往症有無
    # 手術種類
    # がん既往有無
    # がん区分
    # TMN分類
    # がん確定日との期間
    # ほかの診断の確定日との期間

    result_other = []

    # 一意なIDリストを作成する
    ids = df['{0}_id'.format(when)].drop_duplicates()
    # ids = pd.Series(df_ids)

    # IDごとに繰り返す
    ii = 0
    for id_ in ids:
        ii += 1

        # IDが紐づく診断書分ループが必要な処理
        # IDが紐づく診断書を取得する
        df_id = df[df['{0}_id'.format(when)] == id_]

        i = 0
        row_other_sum = []

        print(when, ii, len(ids), id_, len(df_id))

        # 同じIDをもつ診断書データ分繰り返す
        for id_no in df_id['id_no']:

            """ その他（傷病コード、手術コード以外） """
            """ No.2 """
            # 傷病発生日
            date_byomei = process_byodate()

            """ No.5 """
            # 治療期間

            # 手術種類
            has_ope_shu = process_ope_shu(df_id)

            # 手術コードの有無

            """ No.8 """
            # 今回治療悪性新生物
            has_gan_konkai = process_gan_konkai(df)

            # がんの有無

            """ 作成した変数をマージ """
            row_other = []

            i += 1

        # IDが紐づく診断書分まとめて可能な処理
        """ No.3 """
        # 前医有無
        has_zeni = [0 if (df_id['zenumu'] == 0).sum() == 0 else 1]
        """ No.4 """
        # 既往症有無
        has_kiou = [0 if (df_id['kiouumu'] == 0).sum() == 0 else 1]
        """ No.8 """
        # 悪性新生物既往区分
        has_gankiou = [0 if (df_id[TODO] == 0).sum() == 0 else 1]

    # DataFrame化
    df_other = pd.DataFrame(result_other, columns=process_certificate_other_columns(when), index=0)  # idをインデックスに
    # 値がすべて0のカラムを削除する
    df_other = drop_all0_columns(df_other)

    return result_byo_code, result_ope_code, result_other, ids


def process_certificate_dummy(df, list_byo, list_ope):
    df_res_byo = df[['id_', 'no_', 'id_no']]
    df_res_ope = df[['id_', 'no_', 'id_no']]
    df_tmp = df[['id_', 'no_', 'id_no']]

    # 診断書データ（傷病）のダミー変数化
    for code in list_byo:
        df_tmp['code_byo1_{0}'.format(code)] = df['code_byo1'] == code
        df_tmp['code_byo2_{0}'.format(code)] = df['code_byo2'] == code
        df_tmp['code_byo3_{0}'.format(code)] = df['code_byo3'] == code
        df_tmp['code_byo4_{0}'.format(code)] = df['code_byo4'] == code
        df_tmp['code_byo5_{0}'.format(code)] = df['code_byo5'] == code
        df_res_byo['code_byo_{0}'.format(code)] = int(df_tmp['code_byo1_{0}'.format(code)] +
                                                  df_tmp['code_byo2_{0}'.format(code)] +
                                                  df_tmp['code_byo3_{0}'.format(code)] +
                                                  df_tmp['code_byo4_{0}'.format(code)] +
                                                  df_tmp['code_byo5_{0}'.format(code)])
        df_tmp['code_byogen1_{0}'.format(code)] = df['code_byogen1'] == code
        df_tmp['code_byogen2_{0}'.format(code)] = df['code_byogen2'] == code
        df_tmp['code_byogen3_{0}'.format(code)] = df['code_byogen3'] == code
        df_res_byo['code_byogen_{0}'.format(code)] = int(df_tmp['code_byogen1_{0}'.format(code)] +
                                                     df_tmp['code_byogen2_{0}'.format(code)] +
                                                     df_tmp['code_byogen3_{0}'.format(code)])
        df_tmp['code_gappei1_{0}'.format(code)] = df['code_gappei1'] == code
        df_tmp['code_gappei2_{0}'.format(code)] = df['code_gappei2'] == code
        df_tmp['code_gappei3_{0}'.format(code)] = df['code_gappei3'] == code
        df_res_byo['code_gappei_{0}'.format(code)] = int(df_tmp['code_gappei1_{0}'.format(code)] +
                                                     df_tmp['code_gappei2_{0}'.format(code)] +
                                                     df_tmp['code_gappei3_{0}'.format(code)])
        df_res_byo['code_hosya_{0}'.format(code)] = int(df['code_hosya'] == code)
        df_res_byo['code_byori_{0}'.format(code)] = int(df['code_byori'] == code)
        df_res_byo['code_hokabyori_{0}'.format(code)] = int(df['code_hokabyori'] == code)
    # 診断書データ（手術）のダミー変数化
    for code in list_ope:
        df_res_ope['code_gappei_ope_{0}'.format(code)] = int(df['code_gappei_ope'] == code)
        df_tmp['code_ope1_{0}'.format(code)] = df['code_ope1'] == code
        df_tmp['code_ope2_{0}'.format(code)] = df['code_ope2'] == code
        df_tmp['code_ope3_{0}'.format(code)] = df['code_ope3'] == code
        df_tmp['code_ope4_{0}'.format(code)] = df['code_ope4'] == code
        df_tmp['code_ope5_{0}'.format(code)] = df['code_ope5'] == code
        df_res_ope['code_ope_{0}'.format(code)] = int(df_tmp['code_ope1_{0}'.format(code)] +
                                                      df_tmp['code_ope2_{0}'.format(code)] +
                                                      df_tmp['code_ope3_{0}'.format(code)] +
                                                      df_tmp['code_ope4_{0}'.format(code)] +
                                                      df_tmp['code_ope5_{0}'.format(code)])
    df_res_byo = drop_all0_columns(df_res_byo)
    df_res_ope = drop_all0_columns(df_res_ope)

    return df_res_byo, df_res_ope


def process_certificate_dummy_groupid(df):
    ids = df['id_'].drop_duplicates().sort_values

    # 自動的にidがインデックスとなる
    df_res = df.groupby('id_').max()

    return df_res


def process_byodate(byodate):
    byodate = byodate.strip()

    # 7文字または8文字であること
    if len(byodate) == 7:
        # 先頭がH,S,Tのいずれかであり、*が含まれていないこと、4-5文字目が01〜12であること、6-7文字目が01〜31であること
        # 1-2文字目が、先頭がHの場合01〜30であること、先頭がSの場合01〜64であること、先頭がTの場合01〜○○であること

        # 先頭がHの場合1988を足す、先頭がSの場合1925を足す、先頭がTの場合○○○○を足す

        return 0

    elif len(byodate) == 8:
        # *が含まれていないこと、1-4文字目が1901〜2018であること、5-6文字目が01〜12であること、7-8文字目が01〜31であること
        pass

    else:
        return 0


def process_ope(df1, df2, df3, df4, df5, df_idno, when):
    ope_12345 = []

    # 手術１〜５のダミー変数化
    for i in range(1, 6):
        # 今回請求の場合、入院期間内または手術日不明の手術を取得する（←過去請求の退院〜今回請求の入院の手術は無視して良い？）
        if when == 'a':
            if trim('手術日{0}'.format(i)) == '' or \
               trim('手術日{0}'.format(i)) >= 入院日 and trim('手術日{0}'.format(i)) <= 退院日:
                ope_12345.append(手術コード)
            else:
                ope_12345.append(0000)
        # 過去請求の場合、退院日以前または手術日不明の手術を取得する
        elif when == 'b':
            if trim('手術日{0}'.format(i)) == '' or \
               trim('手術日{0}'.format(i)) <= 退院日:
                ope_12345.append(手術コード)
            else:
                ope_12345.append(0000)

    # TODO HOME 手術コードを示すカラム名とコード不可の場合の値を設定
    has_ope = np.array(int(i in ope_12345)
                       for i in list_ope if i != '' and i != TODO)

    return has_ope


def process_ope_shu(df):
    # 手術種類1_2〜5_2のダミー変数化
    # TODO HOME 手術種類のカラム名を設定
    has_ope_shu = np.array([int(i in df[TODO].values or i in df[TODO].values or
                            i in df[TODO].values or i in df[TODO].values or
                            i in df[TODO].values)
                        for i in [i for i in range(1, 16)] if i != '' and i != TODO])

    return has_ope_shu


def process_gan_konkai(df):
    # 今回治療悪性新生物のダミー変数化
    # TODO HOME カラム名を設定
    has_gan_konkai = np.array([int(i in df[TODO].values or i in df[TODO].values or
                            i in df[TODO].values or i in df[TODO].values or
                            i in df[TODO].values)
                        for i in [i for i in range(0, 5)] if i != '' and i != TODO])
    return has_gan_konkai


def process_certificate_other_columns(when):
    when = 'now' if when == 'a' else 'bef'


    return np.r_[None, None]


def drop_all0_columns(df):
    # 値がすべて0のカラムを削除する
    drop_columns = []
    for column in df.columns:
        if df[column].sum() == 0:
            drop_columns.append(column)
    df = df.drop(drop_columns, axis=1)

    return df


if __name__ == "__main__":

    # DB接続情報
    conn = psycopg2.connect('dbname=daido_db host=localhost user=postgres password=postgres')
    cur = conn.cursor()

    # main()

    # cur.close()

    df = pd.DataFrame({'C1': [1, 1, 2, 2, 3, 3],
                       'C2': [4, 5, 4, 0, 0, 0],
                       'C3': [True, False, True, True, False, False]})

    print(df.groupby('C1').max())

    # df2 = df[['C1', 'C5']]
    # print(df2)

    print("Finish!")
    
    