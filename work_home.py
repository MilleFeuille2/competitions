# -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime


num = 100000


def main():
    print(datetime.today())

    """ 今回請求IDと過去請求IDごとの査定データをコード化データ付きで取得する """
    sql_get_satei = get_satei()
    df_satei = pd.read_sql(sql=sql_get_satei, con=conn, index_col=None)

    """ 今回請求IDごとの診断書データをコード化データ付きで取得する """
    sql_get_certificate_now = get_certificate_now()
    df_cer_now = pd.read_sql(sql=sql_get_certificate_now, con=conn, index_col=None)
    """ 過去請求IDごとの請求書データをコード化データ付きで取得する """
    sql_get_certificate_bef = get_certificate_bef()
    df_cer_bef = pd.read_sql(sql=sql_get_certificate_bef, con=conn, index_col=None)

    print(datetime.today())
    # df_satei.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\sql\df_satei.csv')
    # df_cer_now.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\sql\df_cer_now.csv')
    # df_cer_bef.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\sql\df_cer_bef.csv')

    # """ 傷病コードリストと手術コードリストと中分類コードリストを作成する """
    list_byo = make_list_byo(df_satei, df_cer_now, df_cer_bef)
    list_ope = make_list_ope(df_cer_now, df_cer_bef)
    list_byo_chu = make_list_byo_chu(df_satei)
    #
    # # list_byo.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_satei\list_byo.csv')
    # # list_ope.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_satei\list_ope.csv')
    # # list_byo_chu.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_satei\list_byo_chu.csv')
    # #
    # # 査定データの加工処理
    # df_res_satei = process_satei(df_satei, list_byo, list_byo_chu)
    #
    # print(datetime.today(), '査定データの加工完了')
    # df_res_satei.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_satei\df_res_satei.csv')
    df_res_satei = pd.read_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_satei\df_res_satei.csv', index_col=0)

    # 診断書データの加工処理
    df_res_cer_now = process_certificate(df_cer_now, list_byo, list_ope, 'a')
    df_res_cer_bef = process_certificate(df_cer_bef, list_byo, list_ope, 'b')

    # データ加工結果を結合する（紐づく診断書が0件の場合も含む。よって外部結合）
    df_res = pd.merge(df_res_satei, df_res_cer_now, left_on='a_id', right_on='id_', how='left')
    df_res = pd.merge(df_res, df_res_cer_bef, left_on='b_id', right_on='id_', how='left')
    df_res = df_res.fillna(0)

    # # """ 過去と今回の診断書の比較によるデータ加工 """
    # # print('過去と今回の診断書の比較によるデータ加工 開始')
    # # # 過去と今回で別の診断書が紐づく場合に、診断書項目が同じかどうか確認して項目を作成する
    # # df_same_cer = pd.read_sql(sql='select * from same_certificate', con=conn, index_col=None)
    # # df_same_cer.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\sql\df_same_cer.csv')
    # #
    # # df_compare = compare_certificate(df_same_cer,
    # #                                  df_cer_now[['id_', 'byo_date', 'syosin_date', 'gankaku_date', 'hokagankaku_date',
    # #                                              'code_hosya', 'code_byori', 'code_hokabyori', 'id_no',
    # #                                              'code_ope1', 'code_ope2', 'code_ope3', 'code_ope4', 'code_ope5']],
    # #                                  df_cer_bef[['id_', 'byo_date', 'syosin_date', 'gankaku_date', 'hokagankaku_date',
    # #                                              'code_hosya', 'code_byori', 'code_hokabyori', 'id_no',
    # #                                              'code_ope1', 'code_ope2', 'code_ope3', 'code_ope4', 'code_ope5']],
    # #                                  df_satei[['a_id', 'b_id']])
    df_compare = pd.read_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_compare\df_compare.csv',
                             index_col=0)
    print('過去と今回の診断書の比較によるデータ加工 完了')

    # 過去の退院日から今回の入院日までの、入院の有無・手術の有無
    df_between = between_certificate(df_satei[['a_id', 'b_id',
                                               'a_hsplzn_dt', 'a_leaving_dt', 'b_hsplzn_dt', 'b_leaving_dt']],
                                     df_cer_now, df_cer_bef)
    # df_between = pd.read_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_compare\df_between.csv',
    #                          index_col=0)

    # 結合する
    df_res = df_res.sort_values(['a_id', 'b_id']).reset_index(drop=True)
    df_compare = df_compare.sort_values(['a_id', 'b_id']).reset_index(drop=True)
    df_between = df_between.sort_values(['a_id', 'b_id']).reset_index(drop=True)
    print('結合')
    df_res = pd.concat([df_res, df_compare.drop(columns=['a_id', 'b_id'])], axis=1)
    df_res = pd.concat([df_res, df_between.drop(columns=['a_id', 'b_id'])], axis=1)

    print('データ加工すべて完了')
    df_res.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_all\df_res.csv', index=False)


def get_satei():
    # 査定データとコード化データを、今回請求と過去請求それぞれ結合して取得する
    return 'SELECT ' \
           ' tol.*, ' \
           ' grp_main1_now.chu_code chu_a_main1,' \
           ' grp_main2_now.chu_code chu_a_main2,' \
           ' grp_main1_bef.chu_code chu_b_main1,' \
           ' grp_main2_bef.chu_code chu_b_main2 ' \
           'FROM daido_total_opportunity as tol ' \
           'LEFT JOIN group_icd9 grp_main1_now ' \
           'ON grp_main1_now.sho_code = tol.a_main_dis_code1 ' \
           'LEFT JOIN group_icd9 grp_main2_now ' \
           'ON grp_main2_now.sho_code = tol.a_main_dis_code2 ' \
           'LEFT JOIN group_icd9 grp_main1_bef ' \
           'ON grp_main1_bef.sho_code = tol.b_main_dis_code1 ' \
           'LEFT JOIN group_icd9 grp_main2_bef ' \
           'ON grp_main2_bef.sho_code = tol.b_main_dis_code2 ' \
           'ORDER BY tol.a_id, tol.b_id '


def get_certificate_now():
    # 今回請求IDに紐づく診断書データをコード付きで取得する
    return ' select ' \
           '   DISTINCT' \
           '   cer.*, ' \
           '   byo1.totalgroup code_byo1,' \
           '   byo2.totalgroup code_byo2,' \
           '   byo3.totalgroup code_byo3,' \
           '   byo4.totalgroup code_byo4,' \
           '   byo5.totalgroup code_byo5,' \
           '   byogen1.totalgroup code_byogen1,' \
           '   byogen2.totalgroup code_byogen2,' \
           '   byogen3.totalgroup code_byogen3,' \
           '   gappei1.totalgroup code_gappei1,' \
           '   gappei2.totalgroup code_gappei2,' \
           '   gappei3.totalgroup code_gappei3,' \
           '   hosya.totalgroup code_hosya,' \
           '   byori.totalgroup code_byori,' \
           '   hokabyori.totalgroup code_hokabyori,' \
           '   concat(gappei_ope.cptPointCd1, gappei_ope.cptPointCd2) code_gappei_ope,' \
           '   concat(ope1.cptPointCd1, ope1.cptPointCd2) code_ope1,' \
           '   concat(ope2.cptPointCd1, ope2.cptPointCd2) code_ope2,' \
           '   concat(ope3.cptPointCd1, ope3.cptPointCd2) code_ope3,' \
           '   concat(ope4.cptPointCd1, ope4.cptPointCd2) code_ope4,' \
           '   concat(ope5.cptPointCd1, ope5.cptPointCd2) code_ope5' \
           ' from' \
           '   daido_certificate_now as cer' \
           ' LEFT JOIN code_byomei1 as byo1' \
           '   on byo1.id_no = cer.id_no' \
           ' LEFT JOIN code_byomei2 as byo2' \
           '   on byo2.id_no = cer.id_no' \
           ' LEFT JOIN code_byomei3 as byo3' \
           '   on byo3.id_no = cer.id_no' \
           ' LEFT JOIN code_byomei4 as byo4' \
           '   on byo4.id_no = cer.id_no' \
           ' LEFT JOIN code_byomei5 as byo5' \
           '   on byo5.id_no = cer.id_no' \
           ' LEFT JOIN code_byogenin1 as byogen1' \
           '   on byogen1.id_no = cer.id_no' \
           ' LEFT JOIN code_byogenin2 as byogen2' \
           '   on byogen2.id_no = cer.id_no' \
           ' LEFT JOIN code_byogenin3 as byogen3' \
           '   on byogen3.id_no = cer.id_no' \
           ' LEFT JOIN code_gapeisyomei1 as gappei1' \
           '   on gappei1.id_no = cer.id_no' \
           ' LEFT JOIN code_gapeisyomei2 as gappei2' \
           '   on gappei2.id_no = cer.id_no' \
           ' LEFT JOIN code_gapeisyomei3 as gappei3' \
           '   on gappei3.id_no = cer.id_no' \
           ' LEFT JOIN code_hosyabui as hosya' \
           '   on hosya.id_no = cer.id_no' \
           ' LEFT JOIN code_byorisinmei as byori' \
           '   on byori.id_no = cer.id_no' \
           ' LEFT JOIN code_hokabyorisinmei as hokabyori' \
           '   on hokabyori.id_no = cer.id_no' \
           ' LEFT JOIN code_gappei_ope as gappei_ope' \
           '   on gappei_ope.id_no = cer.id_no' \
           ' LEFT JOIN code_ope1 as ope1' \
           '   on ope1.id_no = cer.id_no' \
           ' LEFT JOIN code_ope2 as ope2' \
           '   on ope2.id_no = cer.id_no' \
           ' LEFT JOIN code_ope3 as ope3' \
           '   on ope3.id_no = cer.id_no' \
           ' LEFT JOIN code_ope4 as ope4' \
           '   on ope4.id_no = cer.id_no' \
           ' LEFT JOIN code_ope5 as ope5' \
           '   on ope5.id_no = cer.id_no' \
           ' order by cer.id_no' \
           ' ' \
           ' offset 0 limit {0} '.format(num)


def get_certificate_bef():
    # 過去請求IDに紐づく診断書データをコード付きで取得する
    return ' select ' \
           '   DISTINCT' \
           '   cer.*, '   \
           '   byo1.totalgroup code_byo1,' \
           '   byo2.totalgroup code_byo2,' \
           '   byo3.totalgroup code_byo3,' \
           '   byo4.totalgroup code_byo4,' \
           '   byo5.totalgroup code_byo5,' \
           '   byogen1.totalgroup code_byogen1,' \
           '   byogen2.totalgroup code_byogen2,' \
           '   byogen3.totalgroup code_byogen3,' \
           '   gappei1.totalgroup code_gappei1,' \
           '   gappei2.totalgroup code_gappei2,' \
           '   gappei3.totalgroup code_gappei3,' \
           '   hosya.totalgroup code_hosya,' \
           '   byori.totalgroup code_byori,' \
           '   hokabyori.totalgroup code_hokabyori,' \
           '   concat(gappei_ope.cptPointCd1, gappei_ope.cptPointCd2) code_gappei_ope,' \
           '   concat(ope1.cptPointCd1, ope1.cptPointCd2) code_ope1,' \
           '   concat(ope2.cptPointCd1, ope2.cptPointCd2) code_ope2,' \
           '   concat(ope3.cptPointCd1, ope3.cptPointCd2) code_ope3,' \
           '   concat(ope4.cptPointCd1, ope4.cptPointCd2) code_ope4,' \
           '   concat(ope5.cptPointCd1, ope5.cptPointCd2) code_ope5' \
           ' from  ' \
           '   daido_certificate_bef as cer' \
           ' LEFT JOIN code_byomei1 as byo1' \
           '   on byo1.id_no = cer.id_no' \
           ' LEFT JOIN code_byomei2 as byo2' \
           '   on byo2.id_no = cer.id_no' \
           ' LEFT JOIN code_byomei3 as byo3' \
           '   on byo3.id_no = cer.id_no' \
           ' LEFT JOIN code_byomei4 as byo4' \
           '   on byo4.id_no = cer.id_no' \
           ' LEFT JOIN code_byomei5 as byo5' \
           '   on byo5.id_no = cer.id_no' \
           ' LEFT JOIN code_byogenin1 as byogen1' \
           '   on byogen1.id_no = cer.id_no' \
           ' LEFT JOIN code_byogenin2 as byogen2' \
           '   on byogen2.id_no = cer.id_no' \
           ' LEFT JOIN code_byogenin3 as byogen3' \
           '   on byogen3.id_no = cer.id_no' \
           ' LEFT JOIN code_gapeisyomei1 as gappei1' \
           '   on gappei1.id_no = cer.id_no' \
           ' LEFT JOIN code_gapeisyomei2 as gappei2' \
           '   on gappei2.id_no = cer.id_no' \
           ' LEFT JOIN code_gapeisyomei3 as gappei3' \
           '   on gappei3.id_no = cer.id_no' \
           ' LEFT JOIN code_hosyabui as hosya' \
           '   on hosya.id_no = cer.id_no' \
           ' LEFT JOIN code_byorisinmei as byori' \
           '   on byori.id_no = cer.id_no' \
           ' LEFT JOIN code_hokabyorisinmei as hokabyori' \
           '   on hokabyori.id_no = cer.id_no' \
           ' LEFT JOIN code_gappei_ope as gappei_ope' \
           '   on gappei_ope.id_no = cer.id_no' \
           ' LEFT JOIN code_ope1 as ope1' \
           '   on ope1.id_no = cer.id_no' \
           ' LEFT JOIN code_ope2 as ope2' \
           '   on ope2.id_no = cer.id_no' \
           ' LEFT JOIN code_ope3 as ope3' \
           '   on ope3.id_no = cer.id_no' \
           ' LEFT JOIN code_ope4 as ope4' \
           '   on ope4.id_no = cer.id_no' \
           ' LEFT JOIN code_ope5 as ope5' \
           '   on ope5.id_no = cer.id_no' \
           ' order by cer.id_no' \
           ' ' \
           ' offset 0 limit {0} '.format(num)


def make_list_byo(df_satei, df_cer_now, df_cer_bef):
    # 傷病コードのリストを作成する（査定データ、診断書データ共通）
    list_byo = \
        pd.concat((df_satei['a_main_dis_code1'], df_satei['a_main_dis_code2'],
                   df_satei['a_cause_dis_code1'], df_satei['a_cause_dis_code2'],
                   df_satei['a_gappei_code'], df_satei['a_kiou_code1'], df_satei['a_kiou_code2'],
                   df_satei['b_main_dis_code1'], df_satei['b_main_dis_code2'],
                   df_satei['b_cause_dis_code1'], df_satei['b_cause_dis_code2'],
                   df_satei['b_gappei_code'], df_satei['b_kiou_code1'], df_satei['b_kiou_code2'],
                   df_cer_now['code_byo1'], df_cer_now['code_byo2'], df_cer_now['code_byo3'],
                   df_cer_now['code_byo4'], df_cer_now['code_byo5'],
                   df_cer_now['code_byogen1'], df_cer_now['code_byogen2'], df_cer_now['code_byogen3'],
                   df_cer_now['code_gappei1'], df_cer_now['code_gappei2'], df_cer_now['code_gappei3'],
                   df_cer_now['code_hosya'], df_cer_now['code_byori'], df_cer_now['code_hokabyori'],
                   df_cer_bef['code_byo1'], df_cer_bef['code_byo2'], df_cer_bef['code_byo3'],
                   df_cer_bef['code_byo4'], df_cer_bef['code_byo5'],
                   df_cer_bef['code_byogen1'], df_cer_bef['code_byogen2'], df_cer_bef['code_byogen3'],
                   df_cer_bef['code_gappei1'], df_cer_bef['code_gappei2'], df_cer_bef['code_gappei3'],
                   df_cer_bef['code_hosya'], df_cer_bef['code_byori'], df_cer_bef['code_hokabyori']),
                  axis=0)
    list_byo = list_byo.drop_duplicates().sort_values()
    # ''と'0'を除外する
    list_byo = pd.Series([code for code in list_byo if code != '000' and code is not None and code.strip() != ''])
    return list_byo


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
    # ''と'0'を除外する
    list_ope = pd.Series([code for code in list_ope if code != '000' and code is not None and code.strip() != ''])
    return list_ope


def make_list_byo_chu(df_satei):
    # 傷病コードの中分類のリストを作成する（査定データ、診断書データ共通）
    list_byo_chu = \
        pd.concat((df_satei['chu_a_main1'], df_satei['chu_a_main2'],
                   df_satei['chu_b_main1'], df_satei['chu_b_main2']),
                  axis=0)
    list_byo_chu = list_byo_chu.drop_duplicates().sort_values()
    # ''と'0'を除外する
    list_byo_chu = pd.Series([code for code in list_byo_chu if code != '000' and code is not None and code.strip() != ''])
    return list_byo_chu


def process_satei(df, list_byo, list_byo_chu):

    df_res = df[['a_id', 'a_main_dis_code1', 'a_main_dis_code2',
                 'b_id', 'b_main_dis_code1', 'b_main_dis_code2',
                 'a_tol_flg']]

    # 今回請求と過去請求のコードを比較して変数を作成する
    # 今回主傷病12と過去主傷病12
    df_res['same_amain12_bmain12'] = ((df['a_main_dis_code1'] != '000') & (df['b_main_dis_code1'] != '000') &
                                      (df['a_main_dis_code2'] != '000') & (df['b_main_dis_code2'] != '000') &
                                      ((df['a_main_dis_code1'] == df['b_main_dis_code1']) &
                                       (df['a_main_dis_code1'] == df['b_main_dis_code2']) |
                                       (df['a_main_dis_code2'] == df['b_main_dis_code1']) |
                                       (df['a_main_dis_code1'] == df['b_main_dis_code2']))) * 1
    # 今回主傷病と過去主傷病
    df_res['same_amain_bmain'] = ((df['a_main_dis_code1'] != '000') & (df['b_main_dis_code1'] != '000') &
                                  (df['a_main_dis_code1'] == df['b_main_dis_code1']) |
                                  (df['a_main_dis_code1'] != '000') & (df['b_main_dis_code2'] != '000') &
                                  (df['a_main_dis_code1'] == df['b_main_dis_code2']) |
                                  (df['a_main_dis_code2'] != '000') & (df['b_main_dis_code1'] != '000') &
                                  (df['a_main_dis_code2'] == df['b_main_dis_code1']) |
                                  (df['a_main_dis_code2'] != '000') & (df['b_main_dis_code2'] != '000') &
                                  (df['a_main_dis_code2'] == df['b_main_dis_code2'])) * 1
    # 今回主傷病と過去原因傷病
    df_res['same_amain_bcause'] = ((df['a_main_dis_code1'] != '000') & (df['b_cause_dis_code1'] != '000') &
                                   (df['a_main_dis_code1'] == df['b_cause_dis_code1']) |
                                   (df['a_main_dis_code1'] != '000') & (df['b_cause_dis_code2'] != '000') &
                                   (df['a_main_dis_code1'] == df['b_cause_dis_code2']) |
                                   (df['a_main_dis_code2'] != '000') & (df['b_cause_dis_code1'] != '000') &
                                   (df['a_main_dis_code2'] == df['b_cause_dis_code1']) |
                                   (df['a_main_dis_code2'] != '000') & (df['b_cause_dis_code2'] != '000') &
                                   (df['a_main_dis_code2'] == df['b_cause_dis_code2'])) * 1
    # 今回主傷病と過去合併症
    df_res['same_amain_bgappei'] = ((df['a_main_dis_code1'] != '000') & (df['b_gappei_code'] != '000') &
                                     (df['a_main_dis_code1'] == df['b_gappei_code']) |
                                     (df['a_main_dis_code2'] != '000') & (df['b_gappei_code'] != '000') &
                                     (df['a_main_dis_code2'] == df['b_gappei_code'])) * 1
    # 今回原因傷病と過去主傷病
    df_res['same_acause_bmain'] = ((df['a_cause_dis_code1'] != '000') & (df['b_main_dis_code1'] != '000') &
                                   (df['a_cause_dis_code1'] == df['b_main_dis_code1']) |
                                   (df['a_cause_dis_code1'] != '000') & (df['b_main_dis_code2'] != '000') &
                                   (df['a_cause_dis_code1'] == df['b_main_dis_code2']) |
                                   (df['a_cause_dis_code2'] != '000') & (df['b_main_dis_code1'] != '000') &
                                   (df['a_cause_dis_code2'] == df['b_main_dis_code1']) |
                                   (df['a_cause_dis_code2'] != '000') & (df['b_main_dis_code2'] != '000') &
                                   (df['a_cause_dis_code2'] == df['b_main_dis_code2'])) * 1
    # 今回原因傷病と過去原因傷病
    df_res['same_acause_bcause'] = ((df['a_cause_dis_code1'] != '000') & (df['b_cause_dis_code1'] != '000') &
                                    (df['a_cause_dis_code1'] == df['b_cause_dis_code1']) |
                                    (df['a_cause_dis_code1'] != '000') & (df['b_cause_dis_code2'] != '000') &
                                    (df['a_cause_dis_code1'] == df['b_cause_dis_code2']) |
                                    (df['a_cause_dis_code2'] != '000') & (df['b_cause_dis_code1'] != '000') &
                                    (df['a_cause_dis_code2'] == df['b_cause_dis_code1']) |
                                    (df['a_cause_dis_code2'] != '000') & (df['b_cause_dis_code2'] != '000') &
                                    (df['a_cause_dis_code2'] == df['b_cause_dis_code2'])) * 1
    # 今回原因傷病と過去合併症
    df_res['same_acause_bgappei'] = ((df['a_cause_dis_code1'] != '000') & (df['b_gappei_code'] != '000') &
                                     (df['a_cause_dis_code1'] == df['b_gappei_code']) |
                                     (df['a_cause_dis_code2'] != '000') & (df['b_gappei_code'] != '000') &
                                     (df['a_cause_dis_code2'] == df['b_gappei_code'])) * 1
    # 今回合併症と過去主傷病
    df_res['same_agappei_bmain'] = ((df['a_gappei_code'] != '000') & (df['b_main_dis_code1'] != '000') &
                                    (df['a_gappei_code'] == df['b_main_dis_code1']) |
                                    (df['a_gappei_code'] != '000') & (df['b_main_dis_code2'] != '000') &
                                    (df['a_gappei_code'] == df['b_main_dis_code2'])) * 1
    # 今回合併症と過去原因傷病
    df_res['same_agappei_bcause'] = ((df['a_gappei_code'] != '000') & (df['b_cause_dis_code1'] != '000') &
                                     (df['a_gappei_code'] == df['b_cause_dis_code1']) |
                                     (df['a_gappei_code'] != '000') & (df['b_cause_dis_code2'] != '000') &
                                     (df['a_gappei_code'] == df['b_cause_dis_code2'])) * 1
    # 今回合併症と過去合併症
    df_res['same_agappei_bgappei'] = ((df['a_gappei_code'] != '000') & (df['b_gappei_code'] != '000') &
                                      (df['a_gappei_code'] == df['b_gappei_code'])) * 1
    # 今回既往症と過去原因傷病
    df_res['same_akiou_bcause'] = ((df['a_kiou_code1'] != '000') & (df['b_cause_dis_code1'] != '000') &
                                  (df['a_kiou_code1'] == df['b_cause_dis_code1']) |
                                  (df['a_kiou_code1'] != '000') & (df['b_cause_dis_code2'] != '000') &
                                  (df['a_kiou_code1'] == df['b_cause_dis_code2']) |
                                  (df['a_kiou_code2'] != '000') & (df['b_cause_dis_code1'] != '000') &
                                  (df['a_kiou_code2'] == df['b_cause_dis_code1']) |
                                  (df['a_kiou_code2'] != '000') & (df['b_cause_dis_code2'] != '000') &
                                  (df['a_kiou_code2'] == df['b_cause_dis_code2'])) * 1
    # 今回既往症と過去合併症
    df_res['same_akiou_bgappei'] = ((df['a_kiou_code1'] != '000') & (df['b_gappei_code'] != '000') &
                                    (df['a_kiou_code1'] == df['b_gappei_code']) |
                                    (df['a_kiou_code2'] != '000') & (df['b_gappei_code'] != '000') &
                                    (df['a_kiou_code2'] == df['b_gappei_code'])) * 1

    # 今回請求の傷病コードのダミー変数を作成する
    df_dummy_now = process_satei_dummy(df, list_byo, 'now')

    # 過去請求の傷病コードのダミー変数を作成する
    df_dummy_bef = process_satei_dummy(df, list_byo, 'bef')
    
    # 主傷病を中分類化してダミー変数化
    df_dummy_now_chu = process_satei_dummy_chu(df, list_byo_chu, 'now')
    df_dummy_bef_chu = process_satei_dummy_chu(df, list_byo_chu, 'bef')

    # 今回原因傷病の有無
    df_res['a_byogen_exist'] = ((df['a_cause_dis_code1'] != '') & (df['a_cause_dis_code1'] != '000')) * 1
    # 今回合併症の有無
    df_res['a_gappei_exist'] = ((df['a_gappei_code'] != '') & (df['a_gappei_code'] != '000')) * 1
    # 今回既往症の有無
    df_res['a_kiou_exist'] = ((df['a_kiou_code1'] != '') & (df['a_kiou_code1'] != '000') |
                              (df['a_kiou_code2'] != '') & (df['a_kiou_code2'] != '000')) * 1
    # 過去原因傷病の有無
    df_res['b_byogen_exist'] = ((df['b_cause_dis_code1'] != '') & (df['b_cause_dis_code1'] != '000')) * 1
    # 過去合併症の有無1
    df_res['b_gappei_exist'] = ((df['b_gappei_code'] != '') & (df['b_gappei_code'] != '000')) * 1
    # 過去既往症の有無
    df_res['b_kiou_exist'] = ((df['b_kiou_code1'] != '') & (df['b_kiou_code1'] != '000') |
                              (df['b_kiou_code2'] != '') & (df['b_kiou_code2'] != '000')) * 1

    # 結合する
    df_res = pd.concat([df_res, df_dummy_now.iloc[:, 2:]], axis=1)
    df_res = pd.concat([df_res, df_dummy_now_chu.iloc[:, 2:]], axis=1)
    df_res = pd.concat([df_res, df_dummy_bef.iloc[:, 2:]], axis=1)
    df_res = pd.concat([df_res, df_dummy_bef_chu.iloc[:, 2:]], axis=1)

    return df_res


def process_satei_dummy(df, list_byo, when):
    when2 = 'a' if when == 'now' else 'b'
    df_res = df[['a_id', 'b_id']]

    for code in list_byo:
        print('査定データ（傷病）のダミー変数化', when, code)
        # 査定データのダミー変数化
        df_res['code_main_{0}_{1}'.format(code, when)] = ((df['{0}_main_dis_code1'.format(when2)] == code) |
                                                          (df['{0}_main_dis_code2'.format(when2)] == code)) * 1
        df_res['code_cause_{0}_{1}'.format(code, when)] = ((df['{0}_cause_dis_code1'.format(when2)] == code) |
                                                           (df['{0}_cause_dis_code2'.format(when2)] == code)) * 1
        df_res['code_gappei_{0}_{1}_sa'.format(code, when)] = (df['{0}_gappei_code'.format(when2)] == code) * 1
        df_res['code_kiou_{0}_{1}'.format(code, when)] = ((df['{0}_kiou_code1'.format(when2)] == code) |
                                                          (df['{0}_kiou_code2'.format(when2)] == code)) * 1

    df_res = drop_all0_columns(df_res)
    return df_res


def process_satei_dummy_chu(df, list_byo_chu, when):
    when2 = 'a' if when == 'now' else 'b'
    df_res = df[['a_id', 'b_id']]

    for code in list_byo_chu:
        print('査定データ（傷病中分類）ダミー変数化', when, code)
        # 査定データのダミー変数化
        df_res['chu_main_{0}_{1}'.format(code, when)] = ((df['chu_{0}_main1'.format(when2)] == code) |
                                                         (df['chu_{0}_main2'.format(when2)] == code)) * 1

    df_res = drop_all0_columns(df_res)
    return df_res


def process_certificate(df, list_byo, list_ope, when):
    
    # # # id_no単位でダミー変数化する
    # # df_dummy_byo, df_dummy_ope = process_certificate_dummy(df, list_byo, list_ope, when)
    # # df_dummy_byo.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_certificate\df_dummy_byo_{0}.csv'.format(when))
    # # df_dummy_ope.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_certificate\df_dummy_ope_{0}.csv'.format(when))
    # # print(datetime.today(), 'id_no単位のダミー変数化完了。id単位に加工する', when)
    # #
    # # # ダミー変数化したものをid単位にする
    # # df_dummy_byo = df_dummy_byo.groupby('id_').max().sort_index()
    # # df_dummy_ope = df_dummy_ope.groupby('id_').max().sort_index()
    # # df_dummy_byo.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_certificate\df_dummy_byo_gid_{0}.csv'.format(when))
    # # df_dummy_ope.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_certificate\df_dummy_ope_gid_{0}.csv'.format(when))
    # # print(datetime.today(), 'id単位のダミー変数化完了（傷病・手術）', when)
    # #
    # df_dummy_byo = pd.read_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_certificate\df_dummy_byo_gid_{0}.csv'.format(when), index_col=0)
    # df_dummy_ope = pd.read_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_certificate\df_dummy_ope_gid_{0}.csv'.format(when), index_col=0)
    #
    # # 結合する
    # df_dummy = pd.concat([df_dummy_byo, df_dummy_ope], axis=1)
    # print('診断書データのダミー変数化完了', when)
    # # df_dummy.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_certificate\df_dummy_{0}.csv'.format(when))
    #
    # """ 傷病・手術コードのダミー変数以外の変数を作成する """
    # df_other = pd.DataFrame(None, index=df['id_'].drop_duplicates().sort_values())
    #
    # df_groupbyid = df[['id_', 'no_', 'zenumu', 'kiouumu', 'uiac0209', 'code_hosya',
    #                    'code_ope1', 'code_ope2', 'code_ope3', 'code_ope4', 'code_ope5',
    #                    'code_byori', 'code_hokabyori']].groupby('id_')
    # df_groupbyid_max = df_groupbyid.max().sort_index()
    #
    # # 前医有無
    # df_other['zenumu_{0}'.format(when)] = df_groupbyid_max['zenumu']
    #
    # # 既往症有無
    # df_other['kiouumu_{0}'.format(when)] = df_groupbyid_max['kiouumu']
    #
    # # 手術種類
    # df_other = pd.concat([df_other, process_ope_kubun(df, when)], axis=1)
    #
    # # 悪性新生物既往区分
    # df_other['gankiouumu_{0}'.format(when)] = df_groupbyid_max['uiac0209']
    #
    # # 今回治療悪性新生物
    # df_tmp = df[['id_']]
    # df_tmp['gan_kubun1_{0}'.format(when)] = (df['konkaiakusei'] == '1') * 1
    # df_tmp['gan_kubun2_{0}'.format(when)] = (df['konkaiakusei'] == '2') * 1
    # df_tmp['gan_kubun3_{0}'.format(when)] = (df['konkaiakusei'] == '3') * 1
    # df_tmp['gan_kubun4_{0}'.format(when)] = (df['konkaiakusei'] == '4') * 1
    # df_tmp_groupbyid_max = df_tmp.groupby('id_').max().sort_index()
    # df_other = pd.concat([df_other, df_tmp_groupbyid_max], axis=1)
    #
    # # TMN分類
    # df_tmp = df[['id_']]
    # df_tmp['byorit_{0}'.format(when)] = df['byorit'].apply(convert_byorit)
    # df_tmp['byorin_{0}'.format(when)] = df['byorin'].apply(convert_byorin)
    # df_tmp['byorim_{0}'.format(when)] = df['byorim'].apply(convert_byorim)
    # df_tmp_groupbyid_max = df_tmp.groupby('id_').max().sort_index()
    # df_other = pd.concat([df_other, df_tmp_groupbyid_max], axis=1)
    #
    # df_tmp = df[['id_']]
    # # 手術の有無（手術１～５がコード化されているかいないか）
    # df_tmp['ope_exist_{0}'.format(when)] = ((df['code_ope1'] != '000') & (df['code_ope1'] != '') & (df['code_ope1'].notna()) |
    #                                         (df['code_ope2'] != '000') & (df['code_ope2'] != '') & (df['code_ope2'].notna()) |
    #                                         (df['code_ope3'] != '000') & (df['code_ope3'] != '') & (df['code_ope3'].notna()) |
    #                                         (df['code_ope4'] != '000') & (df['code_ope4'] != '') & (df['code_ope4'].notna()) |
    #                                         (df['code_ope5'] != '000') & (df['code_ope5'] != '') & (df['code_ope5'].notna())) * 1
    # # 放射性部位の有無（コード化されているかいないか）
    # df_tmp['hosyabui_exist_{0}'.format(when)] = ((df['code_hosya'] != '000') & (df['code_hosya'] != '') & (df['code_hosya'].notna())) * 1
    # # がんの有無（病理組織診断名と他の検査による診断名がどちらもコード化されていないもの）
    # df_tmp['gan_exist_{0}'.format(when)] = ((df['code_byori'] != '000') & (df['code_byori'] != '') & (df['code_byori'].notna()) |
    #                                         (df['code_hokabyori'] != '000') & (df['code_hokabyori'] != '') & (df['code_hokabyori'].notna())) * 1
    # df_tmp_groupbyid_max = df_tmp.groupby('id_').max().sort_index()
    # df_other = pd.concat([df_other, df_tmp_groupbyid_max], axis=1)
    # df_other = df_other.fillna(0)
    #
    # # 紐づく診断書の枚数
    # df_tmp = df[['id_']]
    # df_tmp['count_cer_{0}'.format(when)] = (df['no_'].notna()).astype(int)
    # df_tmp_groupbyid_max = df_tmp.groupby('id_').sum().sort_index()
    # df_other = pd.concat([df_other, df_tmp_groupbyid_max], axis=1)
    # df_other = drop_all0_columns(df_other)
    #
    # print('診断書データの他の変数の加工完了', when)
    # df_other.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_certificate\df_other_{0}.csv'.format(when))
    #
    # # ダミー変数と他の変数を結合する
    # df_res = pd.concat([df_dummy, df_other], axis=1)
    #
    # print('診断書データの加工完了', when)
    # df_res.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_certificate\df_res_{0}.csv'.format(when))
    df_res = pd.read_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_certificate\df_res_{0}.csv'.format(when), index_col=0)

    return df_res


def process_certificate_dummy(df, list_byo, list_ope, when):
    when2 = 'now' if when == 'a' else 'bef'
    df_res_byo = df[['id_']]
    df_res_ope = df[['id_']]
    df_tmp = df[['id_']]

    # 診断書データ（傷病）のダミー変数化
    print('診断書データ（傷病）のダミー変数化 開始')
    for code in list_byo:
        df_res_byo['code_byo_{0}_{1}'.format(code, when2)] = ((df['code_byo1'] == code) |
                                                              (df['code_byo2'] == code) |
                                                              (df['code_byo3'] == code) |
                                                              (df['code_byo4'] == code) |
                                                              (df['code_byo5'] == code)) * 1
        df_res_byo['code_byogen_{0}_{1}'.format(code, when2)] = ((df['code_byogen1'] == code) |
                                                                 (df['code_byogen2'] == code) |
                                                                 (df['code_byogen3'] == code)) * 1
        df_res_byo['code_gappei_{0}_{1}'.format(code, when2)] = ((df['code_gappei1'] == code) |
                                                                 (df['code_gappei2'] == code) |
                                                                 (df['code_gappei3'] == code)) * 1
        df_res_byo['code_hosya_{0}_{1}'.format(code, when2)] = (df['code_hosya'] == code) * 1
        df_res_byo['code_byori_{0}_{1}'.format(code, when2)] = (df['code_byori'] == code) * 1
        df_res_byo['code_hokabyori_{0}_{1}'.format(code, when2)] = (df['code_hokabyori'] == code) * 1

    # 診断書データ（手術）のダミー変数化
    print('診断書データ（手術）のダミー変数化 開始')
    for code in list_ope:
        df_res_ope['code_gappei_ope_{0}_{1}'.format(code, when2)] = (df['code_gappei_ope'] == code) * 1
        for i in range(1, 6):
            df_tmp['code_ope{0}_{1}'.format(i, code)] = df['code_ope{0}'.format(i)] == code
        df_res_ope['code_ope_{0}_{1}'.format(code, when2)] = ((df['code_ope1'] == code) |
                                                              (df['code_ope2'] == code) |
                                                              (df['code_ope3'] == code) |
                                                              (df['code_ope4'] == code) |
                                                              (df['code_ope5'] == code)) * 1
    df_res_byo = drop_all0_columns(df_res_byo)
    df_res_ope = drop_all0_columns(df_res_ope)

    return df_res_byo, df_res_ope


def process_ope_kubun(df, when):
    # 手術種類1_1〜5_2のダミー変数化
    df_tmp = df[['id_']]

    for i in range(1, 16):
        df_tmp['ope_shu{0}_{1}'.format(i, when)] = ((df['uiaa0601_1'] == i) |
                                                    (df['uiaa0601_2'] == i) |
                                                    (df['uiaa0601_3'] == i) |
                                                    (df['uiaa0601_4'] == i) |
                                                    (df['uiaa0601_5'] == i) |
                                                    (df['uiaa0601_12'] == i) |
                                                    (df['uiaa0601_22'] == i) |
                                                    (df['uiaa0601_32'] == i) |
                                                    (df['uiaa0601_42'] == i) |
                                                    (df['uiaa0601_52'] == i)) * 1

    df_tmp_groupbyid_max = df_tmp.groupby('id_').max().sort_index()

    return df_tmp_groupbyid_max


def drop_all0_columns(df):
    # 値がすべて0のカラムを削除する
    drop_columns = []
    for column in df.columns:
        if df[column].max() == 0:
            print('値がすべて0のカラム：', column)
            drop_columns.append(column)
    print('値がすべて0のカラム抽出完了。削除に入る')
    df = df.drop(drop_columns, axis=1)

    return df


def convert_date(x):
    # 正しい日付のみ残し、残りは99999999にする
    if x is None:
        return '99999999'

    if len(x) == 7:
        if x[0] == 'H':
            if x[1:3] in ['{0:02d}'.format(i) for i in range(1, 31)] and \
               x[3:5] in ['{0:02d}'.format(i) for i in range(1, 13)] and \
               x[5:] in ['{0:02d}'.format(i) for i in range(1, 32)]:
                return str(int(x[1:3]) + 1988) + x[3:5] + x[5:]

        if x[0] == 'S':
            if x[1:3] in ['{0:02d}'.format(i) for i in range(25, 65)] and \
               x[3:5] in ['{0:02d}'.format(i) for i in range(1, 13)] and \
               x[5:] in ['{0:02d}'.format(i) for i in range(1, 32)]:
                return str(int(x[1:3]) + 1925) + x[3:5] + x[5:]

    if len(x) == 8:
        if x[0:4] in [str(i) for i in range(1950, 2019)] and \
           x[4:6] in ['{0:02d}'.format(i) for i in range(1, 13)] and \
           x[6:] in ['{0:02d}'.format(i) for i in range(1, 32)]:
            return str(x)

    return '99999999'


def convert_byorit(x):
    # TNM分類のTを変換する 0,is(=0.5),1,2,3,4
    if x is None:
        return -1

    if x[:2].lower() == 'is':
        return 0.5

    elif x[:1] in ['0', '1', '2', '3', '4']:
        return int(x[0])

    else:
        return -1


def convert_byorin(x):
    # TNM分類のNを変換する N:0,1,2,3,4
    if x is None:
        return -1

    if x[:1] in ['0', '1', '2', '3', '4']:
        return int(x[0])

    else:
        return -1


def convert_byorim(x):
    # TNM分類のMを変換する M:0,1,2,3
    if x is None:
        return -1

    if x[:1] in ['0', '1', '2', '3']:
        return int(x[0])

    else:
        return -1


def compare_certificate(df_same_cer, df_cer_now, df_cer_bef, df_satei):

    result_compare = []
    for column in ['byo_date', 'syosin_date', 'gankaku_date', 'hokagankaku_date']:
        df_cer_now[column] = df_cer_now[column].apply(convert_date)
        df_cer_bef[column] = df_cer_bef[column].apply(convert_date)

    # 査定データ分くりかえす
    for index in range(len(df_satei)):

        print(index)
        a_id, b_id = df_satei.loc[index, 'a_id'], df_satei.loc[index, 'b_id']

        df_cer_now_ = (df_cer_now[df_cer_now['id_'] == a_id]).drop_duplicates().reset_index(drop=True)
        df_cer_bef_ = (df_cer_bef[df_cer_bef['id_'] == b_id]).drop_duplicates().reset_index(drop=True)

        result_compare.append(pd.concat([pd.Series([a_id, b_id],
                                                   index=['a_id', 'b_id']),
                                         cycle_now_bef(df_same_cer, df_cer_now_, df_cer_bef_)]))

    df_compare = pd.DataFrame(result_compare)
    df_compare.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_compare\df_compare.csv')

    return df_compare


def cycle_now_bef(df_same_cer, df_cer_now_, df_cer_bef_):
    byodate, syosindate, hosya, byori, gankakudate, hokabyori, hokagankakudate, ope = 0, 0, 0, 0, 0, 0, 0, 0

    df_same_cer['now_bef_id_no'] = df_same_cer['now_id_no'].str.cat(df_same_cer['bef_id_no'], sep='-')

    # 紐づく診断書がどちらか0件の場合、何もしない
    if len(df_cer_now_) != 0 and len(df_cer_bef_) != 0 and \
       df_cer_now_.loc[0, 'id_no'] is not None and df_cer_bef_.loc[0, 'id_no'] is not None:

        # 今回請求と過去請求の診断書分それぞれくりかえす
        for index_now in range(len(df_cer_now_)):
            for index_bef in range(len(df_cer_bef_)):

                # 今回と過去が同じ診断書である場合、何もしない
                if df_cer_now_.loc[index_now, 'id_no'] + '-' + df_cer_bef_.loc[index_bef, 'id_no']\
                        in df_same_cer['now_bef_id_no'].values:
                    continue

                # 診断書の各項目が同じかチェック
                row_cer_now_ = df_cer_now_.loc[index_now,
                                               ['byo_date', 'syosin_date', 'gankaku_date', 'hokagankaku_date',
                                                'code_hosya', 'code_byori', 'code_hokabyori',
                                                'code_ope1', 'code_ope2', 'code_ope3', 'code_ope4', 'code_ope5']]
                row_cer_bef_ = df_cer_bef_.loc[index_bef,
                                               ['byo_date', 'syosin_date', 'gankaku_date', 'hokagankaku_date',
                                                'code_hosya', 'code_byori', 'code_hokabyori',
                                                'code_ope1', 'code_ope2', 'code_ope3', 'code_ope4', 'code_ope5']]
                # 傷病発生日
                if row_cer_now_[0] != '99999999' and row_cer_now_[0] == row_cer_bef_[0]:
                    byodate = 1
                # 初診日
                if row_cer_now_[1] != '99999999' and row_cer_now_[1] == row_cer_bef_[1]:
                    syosindate = 1
                # 病理組織診断確定日
                if row_cer_now_[2] != '99999999' and row_cer_now_[2] == row_cer_bef_[2]:
                    gankakudate = 1
                # 他の検査による診断確定日
                if row_cer_now_[3] != '99999999' and row_cer_now_[3] == row_cer_bef_[3]:
                    hokagankakudate = 1
                # 放射性部位
                if row_cer_now_[4] != '000' and row_cer_now_[4] != '' and row_cer_now_[4] is not None and \
                   row_cer_now_[4] == row_cer_bef_[4]:
                    hosya = 1
                # 病理組織診断名
                if row_cer_now_[5] != '000' and row_cer_now_[5] != '' and row_cer_now_[5] is not None and \
                   row_cer_now_[5] == row_cer_bef_[5]:
                    byori = 1
                # 他の検査による診断名
                if row_cer_now_[6] != '000' and row_cer_now_[6] != '' and row_cer_now_[6] is not None and \
                   row_cer_now_[6] == row_cer_bef_[6]:
                    hokabyori = 1
                # 手術名１～５
                if row_cer_now_[7] != 'ZZZZ' and row_cer_now_[7] != '' and row_cer_now_[7] is not None and \
                   row_cer_now_[7] in row_cer_bef_[7:].values:
                    ope = 1
                if row_cer_now_[8] != 'ZZZZ' and row_cer_now_[8] != '' and row_cer_now_[8] is not None and \
                   row_cer_now_[8] in row_cer_bef_[7:].values:
                    ope = 1
                if row_cer_now_[9] != 'ZZZZ' and row_cer_now_[9] != '' and row_cer_now_[9] is not None and \
                   row_cer_now_[9] in row_cer_bef_[7:].values:
                    ope = 1
                if row_cer_now_[10] != 'ZZZZ' and row_cer_now_[10] != '' and row_cer_now_[10] is not None and \
                   row_cer_now_[10] in row_cer_bef_[7:].values:
                    ope = 1
                if row_cer_now_[11] != 'ZZZZ' and row_cer_now_[11] != '' and row_cer_now_[11] is not None and \
                   row_cer_now_[11] in row_cer_bef_[7:].values:
                    ope = 1

    return pd.Series([byodate, syosindate, hosya, byori, gankakudate, hokabyori, hokagankakudate, ope],
                     index=['compare_byodate', 'compare_syosindate', 'compare_hosya', 'compare_byori',
                            'compare_gankakudate', 'compare_hokabyori', 'compare_hokagankakudate', 'compare_ope'])


def between_certificate(df_satei, df_cer_now, df_cer_bef):

    result = []

    for column in ['byo_date', 'syosin_date', 'gankaku_date', 'hokagankaku_date']:
        df_cer_now[column] = df_cer_now[column].apply(convert_date)

    for index in range(len(df_satei)):
        print(index)

        b_nyu = df_satei.loc[index, 'b_hsplzn_dt']
        b_tai = df_satei.loc[index, 'b_leaving_dt']
        a_nyu = df_satei.loc[index, 'a_hsplzn_dt']

        df_now = df_cer_now[df_cer_now['id_'] == df_satei.loc[index, 'a_id']]
        df_bef = df_cer_bef[df_cer_bef['id_'] == df_satei.loc[index, 'b_id']]

        # 過去請求の退院日から今回請求の入院日までの、入院の有無
        if len(df_now) > 0:
            nyuinumu_from_btai_now = (((df_now['nyu1'] >= b_tai) & (df_now['tai1'] <= a_nyu) |
                                       (df_now['nyu2'] >= b_tai) & (df_now['tai2'] <= a_nyu) |
                                       (df_now['nyu3'] >= b_tai) & (df_now['tai3'] <= a_nyu) |
                                       (df_now['nyu4'] >= b_tai) & (df_now['tai4'] <= a_nyu) |
                                       (df_now['nyu5'] >= b_tai) & (df_now['tai5'] <= a_nyu) |
                                       (df_now['nyu6'] >= b_tai) & (df_now['tai6'] <= a_nyu) |
                                       (df_now['nyu7'] >= b_tai) & (df_now['tai7'] <= a_nyu) |
                                       (df_now['nyu8'] >= b_tai) & (df_now['tai8'] <= a_nyu) |
                                       (df_now['nyu9'] >= b_tai) & (df_now['tai9'] <= a_nyu) |
                                       (df_now['nyu10'] >= b_tai) & (df_now['tai10'] <= a_nyu) |
                                       (df_now['nyu11'] >= b_tai) & (df_now['tai11'] <= a_nyu) |
                                       (df_now['nyu12'] >= b_tai) & (df_now['tai12'] <= a_nyu) |
                                       (df_now['nyu13'] >= b_tai) & (df_now['tai13'] <= a_nyu) |
                                       (df_now['nyu14'] >= b_tai) & (df_now['tai14'] <= a_nyu)) * 1).max()
        else:
            nyuinumu_from_btai_now = 0
        if len(df_bef) > 0:
            nyuinumu_from_btai_bef = (((df_bef['nyu1'] >= b_tai) & (df_bef['tai1'] <= a_nyu) |
                                       (df_bef['nyu2'] >= b_tai) & (df_bef['tai2'] <= a_nyu) |
                                       (df_bef['nyu3'] >= b_tai) & (df_bef['tai3'] <= a_nyu) |
                                       (df_bef['nyu4'] >= b_tai) & (df_bef['tai4'] <= a_nyu) |
                                       (df_bef['nyu5'] >= b_tai) & (df_bef['tai5'] <= a_nyu) |
                                       (df_bef['nyu6'] >= b_tai) & (df_bef['tai6'] <= a_nyu) |
                                       (df_bef['nyu7'] >= b_tai) & (df_bef['tai7'] <= a_nyu) |
                                       (df_bef['nyu8'] >= b_tai) & (df_bef['tai8'] <= a_nyu) |
                                       (df_bef['nyu9'] >= b_tai) & (df_bef['tai9'] <= a_nyu) |
                                       (df_bef['nyu10'] >= b_tai) & (df_bef['tai10'] <= a_nyu) |
                                       (df_bef['nyu11'] >= b_tai) & (df_bef['tai11'] <= a_nyu) |
                                       (df_bef['nyu12'] >= b_tai) & (df_bef['tai12'] <= a_nyu) |
                                       (df_bef['nyu13'] >= b_tai) & (df_bef['tai13'] <= a_nyu) |
                                       (df_bef['nyu14'] >= b_tai) & (df_bef['tai14'] <= a_nyu)) * 1).max()
        else:
            nyuinumu_from_btai_bef = 0
        nyuinumu_from_btai = 1 if nyuinumu_from_btai_now + nyuinumu_from_btai_bef >= 1 else 0

        # 過去請求の退院日から今回請求の入院日までの、手術の有無
        if len(df_now) > 0:
            opeumu_from_btai_now = (((df_now['uiaa0611_1'] > b_tai) & (df_now['uiaa0611_1'] < a_nyu) |
                                     (df_now['uiaa0611_2'] > b_tai) & (df_now['uiaa0611_2'] < a_nyu) |
                                     (df_now['uiaa0611_3'] > b_tai) & (df_now['uiaa0611_3'] < a_nyu) |
                                     (df_now['uiaa0611_4'] > b_tai) & (df_now['uiaa0611_4'] < a_nyu) |
                                     (df_now['uiaa0611_5'] > b_tai) & (df_now['uiaa0611_5'] < a_nyu)) * 1).max()
        else:
            opeumu_from_btai_now = 0
        if len(df_bef) > 0:
            opeumu_from_btai_bef = (((df_bef['uiaa0611_1'] > b_tai) & (df_bef['uiaa0611_1'] < a_nyu) |
                                     (df_bef['uiaa0611_2'] > b_tai) & (df_bef['uiaa0611_2'] < a_nyu) |
                                     (df_bef['uiaa0611_3'] > b_tai) & (df_bef['uiaa0611_3'] < a_nyu) |
                                     (df_bef['uiaa0611_4'] > b_tai) & (df_bef['uiaa0611_4'] < a_nyu) |
                                     (df_bef['uiaa0611_5'] > b_tai) & (df_bef['uiaa0611_5'] < a_nyu)) * 1).max()
        else:
            opeumu_from_btai_bef = 0
        opeumu_from_btai = 1 if opeumu_from_btai_now + opeumu_from_btai_bef >= 1 else 0

        result_row = pd.Series([df_satei.loc[index, 'a_id'], df_satei.loc[index, 'b_id'],
                                nyuinumu_from_btai, opeumu_from_btai],
                               index=['a_id', 'b_id', 'nyuinumu_from_btai', 'opeumu_from_btai'])

        result.append(result_row)
        # 今回請求の診断書加工の場合、過去請求の入院日・退院日との期間を計算する
        # 傷病発生日、初診日、病理組織診断確定日、他の検査による診断確定日との期間（値があるもので最長と最短）
        for column in ['byo_date', 'syosin_date', 'gankaku_date', 'hokagankaku_date']:
            if len(df_now) > 0:
                max_nyu = pd.Series([(datetime.strptime(x, "%Y%m%d") - datetime.strptime(b_nyu, "%Y%m%d")).days
                                    if x != '99999999' else -9999999 for x in df_now[column].values]).max()
                min_nyu = pd.Series([(datetime.strptime(x, "%Y%m%d") - datetime.strptime(b_nyu, "%Y%m%d")).days
                                    if x != '99999999' else 99999999 for x in df_now[column].values]).min()
                max_tai = pd.Series([(datetime.strptime(x, "%Y%m%d") - datetime.strptime(b_tai, "%Y%m%d")).days
                                    if x != '99999999' else -9999999 for x in df_now[column].values]).max()
                min_tai = pd.Series([(datetime.strptime(x, "%Y%m%d") - datetime.strptime(b_tai, "%Y%m%d")).days
                                    if x != '99999999' else 99999999 for x in df_now[column].values]).min()
            else:
                max_nyu = -9999999
                min_nyu = 99999999
                max_tai = -9999999
                min_tai = 99999999
            result_row = pd.concat([result_row, pd.Series([max_nyu, min_nyu, max_tai, min_tai],
                                                          index=['{0}_from_bnyu_max'.format(column),
                                                                 '{0}_from_bnyu_min'.format(column),
                                                                 '{0}_from_btai_max'.format(column),
                                                                 '{0}_from_btai_min'.format(column)])])

    df_result = pd.DataFrame(result)
    df_result.to_csv(r'C:\Users\tie303957\PycharmProjects\Ai_suggest\output\processing_compare\df_between.csv')

    return df_result


if __name__ == "__main__":

    print(datetime.today(), 'START')

    # DB接続情報
    conn = psycopg2.connect('dbname=daido_db host=localhost user=postgres password=postgres')
    cur = conn.cursor()

    main()

    cur.close()

    print(datetime.today(), 'END')

    a = pd.Series([1, 2, 3, None])
    # print(a.isnull())
    # print(a.notna())
    #
    # a = True
    # b = False
    # df_test = pd.DataFrame(index=[1,2,3])
    # df_test['column1'] = [a, a, b]
    # df_test['column2'] = [a, b, b]
    #
    # print(df_test)
    # print(df_test['column1'] * 1)
    # print(df_test['column2'] * 1)
    # print((df_test['column1'] & df_test['column2']) * 1)
    # print((df_test['column1'] | df_test['column2']) * 1)
    #
    # # c = pd.Series([1, 1, 0, 0])
    # # d = pd.Series([1, 0, 1, 0])
    # print(c)
    #
    # c_max = c.groupby('b').max()
    # print(c_max)
    #
    # c_min = c.groupby('b').min()
    # print(c_min)




