add file 2.joblib;
add file projects/2/model.py;
add file projects/2/predict.py;

INSERT INTO TABLE hw2_pred
SELECT TRANSFORM(*)
USING 'predict.py'
AS (id, pred)
FROM (
    SELECT *
    FROM hw2_test
    WHERE if1 > 20 and if1 < 40
) as features;
