-- ShioRIS3で計算された線量データの重複レコードをクリーンアップ
-- 実行前にバックアップを取ることを推奨

-- 1. 計算済み線量に関連するfilesレコードを削除
DELETE FROM files
WHERE study_id IN (
    SELECT id FROM studies
    WHERE modality='RTDOSE'
    AND (study_name='ShioRIS3 Calculated Dose' OR study_name LIKE '%Calculated%')
);

-- 2. dose_volumesレコードを削除
DELETE FROM dose_volumes
WHERE study_id IN (
    SELECT id FROM studies
    WHERE modality='RTDOSE'
    AND (study_name='ShioRIS3 Calculated Dose' OR study_name LIKE '%Calculated%')
);

-- 3. studiesレコードを削除
DELETE FROM studies
WHERE modality='RTDOSE'
AND (study_name='ShioRIS3 Calculated Dose' OR study_name LIKE '%Calculated%');

-- 確認用クエリ
SELECT 'Cleanup completed. Remaining studies:' as message;
SELECT id, patient_key, modality, study_name FROM studies WHERE modality='RTDOSE';
