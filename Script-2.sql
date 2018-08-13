


-- 入院期間1〜14がすべてブランクまたは査定データの入院期間と1日でも重なることを条件とし、
-- 過去請求のID、入院日、退院日と診断書データを紐づけて取得する
select distinct
            tol.b_id id_,
            tol.b_hsplzn_dt hsplzn_dt,
            tol.b_leaving_dt leaving_dt,
            b.ent_date, 
            b.sinseibetu, 
            b.tanjo_date_y, 
            b.byomei, 
            b.byo_date, 
            b.suitei1, 
            b.byogenin, 
            b.byogenin_date, 
            b.suitei2, 
            b.gapeisyomei, 
            b.gapeisyo_date, 
            b.suitei3, 
            b.syosin_date, 
            b.syusin_date, 
            b.genjyou1, 
            b.uiaa0511_01 nyu1, 
            b.uiaa0512_01 tai1, 
            b.uiaa0503_01, 
            b.uiaa0511_02 nyu2, 
            b.uiaa0512_02 tai2, 
            b.uiaa0503_02, 
            b.zenumu, 
            b.kiouumu, 
            b.gsbyomei, 
            b.gnyuiinumu, 
            b.gtiryos_date, 
            b.gtiryoe_date, 
            b.gtiryonaiyo, 
            b.byorisinmei, 
            b.byorit, 
            b.byorin, 
            b.byorim, 
            b.gankaku_date, 
            b.jyohiganumu, 
            b.uiaa0601_1, 
            b.syujyumei1, 
            b.kj1, 
            b.kjsyousai1, 
            b.uiaa0611_1, 
            b.uiaa0618_1, 
            b.uiac0216_1, 
            b.uiac0217_1, 
            b.uiac0218_1, 
            b.uiaa0601_2, 
            b.syujyumei2, 
            b.kj2, 
            b.kjsyousai2, 
            b.uiaa0611_2, 
            b.uiaa0618_2, 
            b.uiac0216_2, 
            b.uiac0217_2, 
            b.uiac0218_2, 
            b.hosyabui, 
            b.hosyas_date, 
            b.hosyae_date, 
            b.hosyaryou, 
            b.sensinnumu, 
            b.isi, 
            b.uiaa0511_03 nyu3, 
            b.uiaa0512_03 tai3, 
            b.uiaa0503_03, 
            b.uiaa0511_04 nyu4, 
            b.uiaa0512_04 tai4, 
            b.uiaa0503_04, 
            b.uiaa0511_05 nyu5, 
            b.uiaa0512_05 tai5, 
            b.uiaa0503_05, 
            b.uiaa0511_06 nyu6, 
            b.uiaa0512_06 tai6, 
            b.uiaa0503_06, 
            b.uiaa0511_07 nyu7, 
            b.uiaa0512_07 tai7, 
            b.uiaa0503_07, 
            b.UIAA0511_08 nyu8, 
            b.UIAA0512_08 tai8, 
            b.UIAA0503_08, 
            b.UIAA0511_09 nyu9, 
            b.UIAA0512_09 tai9, 
            b.UIAA0503_09, 
            b.UIAA0511_10 nyu10, 
            b.UIAA0512_10 tai10, 
            b.UIAA0503_10, 
            b.UIAA0511_11 nyu11, 
            b.UIAA0512_11 tai11, 
            b.UIAA0503_11, 
            b.UIAA0511_12 nyu12, 
            b.UIAA0512_12 tai12, 
            b.UIAA0503_12, 
            b.uiaa0511_13 nyu13, 
            b.uiaa0512_13 tai13, 
            b.uiaa0503_13, 
            b.uiaa0511_14 nyu14, 
            b.uiaa0512_14 tai14, 
            b.uiaa0503_14, 
            b.uiac0206, 
            b.uiac0201, 
            b.uiac0202, 
            b.uiac0203, 
            b.uiac0204, 
            b.uiac0205, 
            b.uiaa0632_11, 
            b.uiaa0632_21, 
            b.uiac0216_11, 
            b.uiac0216_12, 
            b.uiaa0632_12, 
            b.uiaa0632_22, 
            b.uiac0216_21, 
            b.uiac0216_22, 
            b.uiaa0632_13, 
            b.uiaa0632_23, 
            b.syujyumei3, 
            b.uiaa0611_3, 
            b.uiaa0601_3, 
            b.uiac0216_31, 
            b.uiac0216_32, 
            b.uiaa0618_3, 
            b.uiac0216_3, 
            b.uiac0217_3, 
            b.uiac0218_3, 
            b.uiaa0632_14, 
            b.uiaa0632_24, 
            b.syujyumei4, 
            b.uiaa0611_4, 
            b.uiaa0601_4, 
            b.uiac0216_41, 
            b.uiac0216_42, 
            b.uiaa0618_4, 
            b.uiac0216_4, 
            b.uiac0217_4, 
            b.uiac0218_4, 
            b.uiaa0632_15, 
            b.uiaa0632_25, 
            b.syujyumei5, 
            b.uiaa0611_5, 
            b.uiaa0601_5, 
            b.uiac0216_51, 
            b.uiac0216_52, 
            b.uiaa0618_5, 
            b.uiac0216_5, 
            b.uiac0217_5, 
            b.uiac0218_5, 
            b.uiac0207, 
            b.uifa0112, 
            b.uiac0208, 
            b.uiac0209, 
            b.uiac0210, 
            b.uiac0211, 
            b.uiac0212, 
            b.uiac0213, 
            b.uifa0114, 
            b.uiac0214, 
            b.uiac0215, 
            b.byomei2, 
            b.byomei3, 
            b.byomei4, 
            b.byomei5, 
            b.byogenin2, 
            b.byogenin3, 
            b.gapeisyomei2, 
            b.gapeisyomei3, 
            b.kioumei1, 
            b.kioumei2, 
            b.hosyasyurui, 
            b.hosyasonota, 
            b.konkaiakusei, 
            b.sensingimei, 
            b.sensin_date, 
            b.sensingiryo, 
            b.uiaa0601_12, 
            b.uiaa0601_22, 
            b.uiaa0601_32, 
            b.uiaa0601_42, 
            b.uiaa0601_52, 
            b.kinkenjin, 
            b.sintatudo, 
            b.sindangazo, 
            b.sindannaisikyo, 
            b.sindansonota, 
            b.hokabyorisinmei, 
            b.hokagankaku_date, 
            b.id_no
            from 
            daido_total_opportunity tol
            left join
            daido_certificate b
            on
            tol.b_id = b.id_
            where
            ((trim(nyu1) = '' or trim(tai1) = '') and (trim(nyu2) = '' or trim(tai2) = '') and
             (trim(nyu3) = '' or trim(tai3) = '') and (trim(nyu4) = '' or trim(tai4) = '') and
             (trim(nyu5) = '' or trim(tai5) = '') and (trim(nyu6) = '' or trim(tai6) = '') and
             (trim(nyu7) = '' or trim(tai7) = '') and (trim(nyu8) = '' or trim(tai8) = '') and
             (trim(nyu9) = '' or trim(tai9) = '') and (trim(nyu10) = '' or trim(tai10) = '') and
             (trim(nyu11) = '' or trim(tai11) = '') and (trim(nyu12) = '' or trim(tai12) = '') and
             (trim(nyu13) = '' or trim(tai13) = '') and (trim(nyu14) = '' or trim(tai14) = '') or
            or
            (case when trim(nyu1) = '' then '99999999' else trim(nyu1) end <= b_leaving_dt and
             case when trim(tai1) = '' then '00000000' else trim(tai1) end >= b_hsplzn_dt)
            or
            (case when trim(nyu2) = '' then '99999999' else trim(nyu2) end <= b_leaving_dt and
             case when trim(tai2) = '' then '00000000' else trim(tai2) end >= b_hsplzn_dt)
            or
            (case when trim(nyu3) = '' then '99999999' else trim(nyu3) end <= b_leaving_dt and
             case when trim(tai3) = '' then '00000000' else trim(tai3) end >= b_hsplzn_dt)
            or
            (case when trim(nyu4) = '' then '99999999' else trim(nyu4) end <= b_leaving_dt and
             case when trim(tai4) = '' then '00000000' else trim(tai4) end >= b_hsplzn_dt)
            or
            (case when trim(nyu5) = '' then '99999999' else trim(nyu5) end <= b_leaving_dt and
             case when trim(tai5) = '' then '00000000' else trim(tai5) end >= b_hsplzn_dt)
            or
            (case when trim(nyu6) = '' then '99999999' else trim(nyu6) end <= b_leaving_dt and
             case when trim(tai6) = '' then '00000000' else trim(tai6) end >= b_hsplzn_dt)
            or
            (case when trim(nyu7) = '' then '99999999' else trim(nyu7) end <= b_leaving_dt and
             case when trim(tai7) = '' then '00000000' else trim(tai7) end >= b_hsplzn_dt)
            or
            (case when trim(nyu8) = '' then '99999999' else trim(nyu8) end <= b_leaving_dt and
             case when trim(tai8) = '' then '00000000' else trim(tai8) end >= b_hsplzn_dt)
            or
            (case when trim(nyu9) = '' then '99999999' else trim(nyu9) end <= b_leaving_dt and
             case when trim(tai9) = '' then '00000000' else trim(tai9) end >= b_hsplzn_dt)
            or
            (case when trim(nyu10) = '' then '99999999' else trim(nyu10) end <= b_leaving_dt and
             case when trim(tai10) = '' then '00000000' else trim(tai10) end >= b_hsplzn_dt)
            or
            (case when trim(nyu11) = '' then '99999999' else trim(nyu11) end <= b_leaving_dt and
             case when trim(tai11) = '' then '00000000' else trim(tai11) end >= b_hsplzn_dt)
            or
            (case when trim(nyu12) = '' then '99999999' else trim(nyu12) end <= b_leaving_dt and
             case when trim(tai12) = '' then '00000000' else trim(tai12) end >= b_hsplzn_dt)
            or
            (case when trim(nyu13) = '' then '99999999' else trim(nyu13) end <= b_leaving_dt and
             case when trim(tai13) = '' then '00000000' else trim(tai13) end >= b_hsplzn_dt)
            or
            (case when trim(nyu14) = '' then '99999999' else trim(nyu14) end <= b_leaving_dt and
             case when trim(tai14) = '' then '00000000' else trim(tai14) end >= b_hsplzn_dt)
            order by b_id
            
            