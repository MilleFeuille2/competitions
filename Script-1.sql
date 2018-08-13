
-- 紐づく診断書が0件の場合は、傷病のみで判断している認識でよいか？

-- 今回傷病と過去傷病ごとの通算機会を取得する
-- 紐づく診断書が0件のものも含むVer
SELECT now, bef, count(*)
FROM (
	SELECT a_main_code1 now, b_main_code1 bef, a_tol_flg flg
	FROM daido_total_opportunity A
	UNION ALL
	SELECT a_main_code1 now, b_main_code2 bef, a_tol_flg flg
	FROM daido_total_opportunity A
	UNION ALL
	SELECT a_main_code2 now, b_main_code1 bef, a_tol_flg flg
	FROM daido_total_opportunity A
	UNION ALL
	SELECT a_main_code2 now, b_main_code2 bef, a_tol_flg flg
	FROM daido_total_opportunity A
) AS tmp
WHERE now != '0000' AND bef != '0000'
GROUP BY now, bef
ORDER BY now, bef


