


-- 入院期間1〜14がすべてブランクまたは査定データの入院期間と1日でも重なることを条件とし、
-- 今回請求のID、入院日、退院日と診断書データを紐づけて取得する
select distinct
            tol.a_id id_,
            tol.a_hsplzn_dt hsplzn_dt,
            tol.a_leaving_dt leaving_dt,
            a.ent_date, 
            a.sinseibetu, 
            a.tanjo_date_y, 
            a.byomei, 
            a.byo_date, 
            a.suitei1, 
            a.byogenin, 
            a.byogenin_date, 
            a.suitei2, 
            a.gapeisyomei, 
            a.gapeisyo_date, 
            a.suitei3, 
            a.syosin_date, 
            a.syusin_date, 
            a.genjyou1, 
            a.uiaa0511_01 nyu1, 
            a.uiaa0512_01 tai1, 
            a.uiaa0503_01, 
            a.uiaa0511_02 nyu2, 
            a.uiaa0512_02 tai2, 
            a.uiaa0503_02, 
            a.zenumu, 
            a.kiouumu, 
            a.gsbyomei, 
            a.gnyuiinumu, 
            a.gtiryos_date, 
            a.gtiryoe_date, 
            a.gtiryonaiyo, 
            a.byorisinmei, 
            a.byorit, 
            a.byorin, 
            a.byorim, 
            a.gankaku_date, 
            a.jyohiganumu, 
            a.uiaa0601_1, 
            a.syujyumei1, 
            a.kj1, 
            a.kjsyousai1, 
            a.uiaa0611_1, 
            a.uiaa0618_1, 
            a.uiac0216_1, 
            a.uiac0217_1, 
            a.uiac0218_1, 
            a.uiaa0601_2, 
            a.syujyumei2, 
            a.kj2, 
            a.kjsyousai2, 
            a.uiaa0611_2, 
            a.uiaa0618_2, 
            a.uiac0216_2, 
            a.uiac0217_2, 
            a.uiac0218_2, 
            a.hosyabui, 
            a.hosyas_date, 
            a.hosyae_date, 
            a.hosyaryou, 
            a.sensinnumu, 
            a.isi, 
            a.uiaa0511_03 nyu3, 
            a.uiaa0512_03 tai3, 
            a.uiaa0503_03, 
            a.uiaa0511_04 nyu4, 
            a.uiaa0512_04 tai4, 
            a.uiaa0503_04, 
            a.uiaa0511_05 nyu5, 
            a.uiaa0512_05 tai5, 
            a.uiaa0503_05, 
            a.uiaa0511_06 nyu6, 
            a.uiaa0512_06 tai6, 
            a.uiaa0503_06, 
            a.uiaa0511_07 nyu7, 
            a.uiaa0512_07 tai7, 
            a.uiaa0503_07, 
            a.UIAA0511_08 nyu8, 
            a.UIAA0512_08 tai8, 
            a.UIAA0503_08, 
            a.UIAA0511_09 nyu9, 
            a.UIAA0512_09 tai9, 
            a.UIAA0503_09, 
            a.UIAA0511_10 nyu10, 
            a.UIAA0512_10 tai10, 
            a.UIAA0503_10, 
            a.UIAA0511_11 nyu11, 
            a.UIAA0512_11 tai11, 
            a.UIAA0503_11, 
            a.UIAA0511_12 nyu12, 
            a.UIAA0512_12 tai12, 
            a.UIAA0503_12, 
            a.uiaa0511_13 nyu13, 
            a.uiaa0512_13 tai13, 
            a.uiaa0503_13, 
            a.uiaa0511_14 nyu14, 
            a.uiaa0512_14 tai14, 
            a.uiaa0503_14, 
            a.uiac0206, 
            a.uiac0201, 
            a.uiac0202, 
            a.uiac0203, 
            a.uiac0204, 
            a.uiac0205, 
            a.uiaa0632_11, 
            a.uiaa0632_21, 
            a.uiac0216_11, 
            a.uiac0216_12, 
            a.uiaa0632_12, 
            a.uiaa0632_22, 
            a.uiac0216_21, 
            a.uiac0216_22, 
            a.uiaa0632_13, 
            a.uiaa0632_23, 
            a.syujyumei3, 
            a.uiaa0611_3, 
            a.uiaa0601_3, 
            a.uiac0216_31, 
            a.uiac0216_32, 
            a.uiaa0618_3, 
            a.uiac0216_3, 
            a.uiac0217_3, 
            a.uiac0218_3, 
            a.uiaa0632_14, 
            a.uiaa0632_24, 
            a.syujyumei4, 
            a.uiaa0611_4, 
            a.uiaa0601_4, 
            a.uiac0216_41, 
            a.uiac0216_42, 
            a.uiaa0618_4, 
            a.uiac0216_4, 
            a.uiac0217_4, 
            a.uiac0218_4, 
            a.uiaa0632_15, 
            a.uiaa0632_25, 
            a.syujyumei5, 
            a.uiaa0611_5, 
            a.uiaa0601_5, 
            a.uiac0216_51, 
            a.uiac0216_52, 
            a.uiaa0618_5, 
            a.uiac0216_5, 
            a.uiac0217_5, 
            a.uiac0218_5, 
            a.uiac0207, 
            a.uifa0112, 
            a.uiac0208, 
            a.uiac0209, 
            a.uiac0210, 
            a.uiac0211, 
            a.uiac0212, 
            a.uiac0213, 
            a.uifa0114, 
            a.uiac0214, 
            a.uiac0215, 
            a.byomei2, 
            a.byomei3, 
            a.byomei4, 
            a.byomei5, 
            a.byogenin2, 
            a.byogenin3, 
            a.gapeisyomei2, 
            a.gapeisyomei3, 
            a.kioumei1, 
            a.kioumei2, 
            a.hosyasyurui, 
            a.hosyasonota, 
            a.konkaiakusei, 
            a.sensingimei, 
            a.sensin_date, 
            a.sensingiryo, 
            a.uiaa0601_12, 
            a.uiaa0601_22, 
            a.uiaa0601_32, 
            a.uiaa0601_42, 
            a.uiaa0601_52, 
            a.kinkenjin, 
            a.sintatudo, 
            a.sindangazo, 
            a.sindannaisikyo, 
            a.sindansonota, 
            a.hokabyorisinmei, 
            a.hokagankaku_date, 
            a.id_no
            from 
            daido_total_opportunity tol
            left join
            daido_certificate b
            on
            tol.a_id = a.id_
            where
            ((trim(nyu1) = '' or trim(tai1) = '') and (trim(nyu2) = '' or trim(tai2) = '') and
             (trim(nyu3) = '' or trim(tai3) = '') and (trim(nyu4) = '' or trim(tai4) = '') and
             (trim(nyu5) = '' or trim(tai5) = '') and (trim(nyu6) = '' or trim(tai6) = '') and
             (trim(nyu7) = '' or trim(tai7) = '') and (trim(nyu8) = '' or trim(tai8) = '') and
             (trim(nyu9) = '' or trim(tai9) = '') and (trim(nyu10) = '' or trim(tai10) = '') and
             (trim(nyu11) = '' or trim(tai11) = '') and (trim(nyu12) = '' or trim(tai12) = '') and
             (trim(nyu13) = '' or trim(tai13) = '') and (trim(nyu14) = '' or trim(tai14) = '') or
            or
            (case when trim(nyu1) = '' then '99999999' else trim(nyu1) end <= a_leaving_dt and
             case when trim(tai1) = '' then '00000000' else trim(tai1) end >= a_hsplzn_dt)
            or
            (case when trim(nyu2) = '' then '99999999' else trim(nyu2) end <= a_leaving_dt and
             case when trim(tai2) = '' then '00000000' else trim(tai2) end >= a_hsplzn_dt)
            or
            (case when trim(nyu3) = '' then '99999999' else trim(nyu3) end <= a_leaving_dt and
             case when trim(tai3) = '' then '00000000' else trim(tai3) end >= a_hsplzn_dt)
            or
            (case when trim(nyu4) = '' then '99999999' else trim(nyu4) end <= a_leaving_dt and
             case when trim(tai4) = '' then '00000000' else trim(tai4) end >= a_hsplzn_dt)
            or
            (case when trim(nyu5) = '' then '99999999' else trim(nyu5) end <= a_leaving_dt and
             case when trim(tai5) = '' then '00000000' else trim(tai5) end >= a_hsplzn_dt)
            or
            (case when trim(nyu6) = '' then '99999999' else trim(nyu6) end <= a_leaving_dt and
             case when trim(tai6) = '' then '00000000' else trim(tai6) end >= a_hsplzn_dt)
            or
            (case when trim(nyu7) = '' then '99999999' else trim(nyu7) end <= a_leaving_dt and
             case when trim(tai7) = '' then '00000000' else trim(tai7) end >= a_hsplzn_dt)
            or
            (case when trim(nyu8) = '' then '99999999' else trim(nyu8) end <= a_leaving_dt and
             case when trim(tai8) = '' then '00000000' else trim(tai8) end >= a_hsplzn_dt)
            or
            (case when trim(nyu9) = '' then '99999999' else trim(nyu9) end <= a_leaving_dt and
             case when trim(tai9) = '' then '00000000' else trim(tai9) end >= a_hsplzn_dt)
            or
            (case when trim(nyu10) = '' then '99999999' else trim(nyu10) end <= a_leaving_dt and
             case when trim(tai10) = '' then '00000000' else trim(tai10) end >= a_hsplzn_dt)
            or
            (case when trim(nyu11) = '' then '99999999' else trim(nyu11) end <= a_leaving_dt and
             case when trim(tai11) = '' then '00000000' else trim(tai11) end >= a_hsplzn_dt)
            or
            (case when trim(nyu12) = '' then '99999999' else trim(nyu12) end <= a_leaving_dt and
             case when trim(tai12) = '' then '00000000' else trim(tai12) end >= a_hsplzn_dt)
            or
            (case when trim(nyu13) = '' then '99999999' else trim(nyu13) end <= a_leaving_dt and
             case when trim(tai13) = '' then '00000000' else trim(tai13) end >= a_hsplzn_dt)
            or
            (case when trim(nyu14) = '' then '99999999' else trim(nyu14) end <= a_leaving_dt and
             case when trim(tai14) = '' then '00000000' else trim(tai14) end >= a_hsplzn_dt)
            order by a_id
            
            
            
            