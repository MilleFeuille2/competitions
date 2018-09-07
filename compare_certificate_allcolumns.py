# -*- encoding:utf-8 -*-
import csv
import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
from datetime import datetime


def main():
    print(datetime.today())

    """ 今回請求IDごとの診断書データを取得する """
    sql_get_certificate_now = get_certificate_now()
    df_cer_now = pd.read_sql(sql=sql_get_certificate_now, con=conn, index_col=None)
    """ 過去請求IDごとの請求書データを取得する """
    sql_get_certificate_bef = get_certificate_bef()
    df_cer_bef = pd.read_sql(sql=sql_get_certificate_bef, con=conn, index_col=None)
    """ 査定データの今回請求ID、過去せきゅうIDを取得する """
    sql_get_satei = get_satei()
    df_satei = pd.read_sql(sql=sql_get_satei, con=conn, index_col=None)

    # 比較する項目のみとする
    # compare_bef = df_cer_bef.drop(columns=['id_', 'no_', 'id_no', 'hsplzn_dt', 'leaving_dt'], axis=1)
    # compare_now = df_cer_now.drop(columns=['id_', 'no_', 'id_no', 'hsplzn_dt', 'leaving_dt'], axis=1)

    result = []  # 結果格納用
    for k in range(len(df_satei)):
        compare_now = df_cer_now[df_cer_now['id_'] == df_satei.loc[k, 'a_id']]
        compare_now_drop = compare_now.drop(columns=['id_', 'no_', 'id_no', 'hsplzn_dt', 'leaving_dt'], axis=1)
        compare_bef = df_cer_bef[df_cer_bef['id_'] == df_satei.loc[k, 'b_id']]
        compare_bef_drop = compare_bef.drop(columns=['id_', 'no_', 'id_no', 'hsplzn_dt', 'leaving_dt'], axis=1)

        # 同じ診断書である過去と今回のid_noの組み合わせを抽出する
        for i in range(len(compare_now_drop)):
            for j in range(len(compare_bef_drop)):
                compare = True

                for num_column in range(len(compare_now_drop.columns)):

                    if compare_now_drop.iloc[i, num_column] != compare_bef_drop.iloc[j, num_column]:
                        compare = False
                        break

                if compare:
                    result.append([compare_now.iloc[i, :]['id_no'], compare_bef.iloc[j, :]['id_no']])
                    print([compare_now.iloc[i, :]['id_no'], compare_bef.iloc[j, :]['id_no']])

    df_result = pd.DataFrame(result)
    df_result.columns = ['now_id_no', 'bef_id_no']

    df_result.to_csv('../same_certificate_allcolumns.csv')

    return None


def get_certificate_now():
    # 今回請求IDに紐づく診断書データをコード付きで取得する
    # 査定テーブルから今回請求と過去請求の入院日・退院日も取得する
    return ' select ' \
           '   cer.* ' \
           ' from' \
           '   daido_certificate_now as cer' \
           ' order by cer.id_no'


def get_certificate_bef():
    # 過去請求IDに紐づく診断書データをコード付きで取得する
    # 査定テーブルから今回請求と過去請求の入院日・退院日も取得する
    return ' select ' \
           '   cer.* '   \
           ' from  ' \
           '   daido_certificate_bef as cer' \
           ' order by cer.id_no'


def get_satei():
    # 過去請求IDに紐づく診断書データをコード付きで取得する
    # 査定テーブルから今回請求と過去請求の入院日・退院日も取得する
    return ' select ' \
           '   tol.a_id, tol.b_id '   \
           ' from  ' \
           '   daido_total_opportunity as tol' \
           ' order by tol.a_id, tol.b_id'


if __name__ == "__main__":

    print(datetime.today(), 'START')

    # DB接続情報
    conn = psycopg2.connect('dbname=daido_db host=localhost user=postgres password=postgres')
    cur = conn.cursor()

    main()

    cur.close()

    print(datetime.today(), 'END')


