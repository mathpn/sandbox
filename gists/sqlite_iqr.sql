/* 
Find and remove outliers per group using the interquartile range with SQLite queries.
*/

-- create table with fake data
CREATE TEMPORARY TABLE test_table (group_id INTEGER, variable REAL);
INSERT INTO test_table
VALUES
    (1, 2),
    (1, 4),
    (1, 4.5),
    (1, 5),
    (1, 20), -- outlier of group 1
    (2, 55), -- outlier of group 2
    (2, 660),
    (2, 690),
    (2, 700),
    (2, 710),
    (2, 711);

-- now the outlier detection part
WITH tb_partition AS (
    SELECT
        group_id,
        variable,
        rn
    FROM (
        SELECT
            group_id,
            variable,
            row_number()
        OVER (
            PARTITION BY group_id
            ORDER BY group_id, variable
        ) AS rn
        FROM test_table
    ) AS x
),
tb_count AS (
    SELECT
        group_id,
        max(rn) AS max_rn
    FROM tb_partition
    GROUP BY group_id
),
tb_75p AS (
    SELECT
        t1.group_id,
        sum(CASE WHEN t1.rn = round(0.75 * t2.max_rn) THEN variable ELSE 0 END) AS variable_75p
    FROM tb_partition AS t1
    LEFT JOIN tb_count AS t2
    ON t1.group_id = t2.group_id
    GROUP BY t1.group_id
),
tb_50p AS (
    SELECT
        t1.group_id,
        sum(CASE WHEN t1.rn = round(0.50 * t2.max_rn) THEN variable ELSE 0 END) AS variable_50p
    FROM tb_partition AS t1
    LEFT JOIN tb_count AS t2
    ON t1.group_id = t2.group_id
    GROUP BY t1.group_id
),
tb_25p AS (
    SELECT
        t1.group_id,
        sum(CASE WHEN t1.rn = round(0.25 * t2.max_rn) THEN variable ELSE 0 END) AS variable_25p
    FROM tb_partition AS t1
    LEFT JOIN tb_count AS t2
    ON t1.group_id = t2.group_id
    GROUP BY t1.group_id
),
tb_bounds AS (
    SELECT
        t1.group_id,
        t1.variable_50p - 2.5 * (t3.variable_75p - t2.variable_25p) AS lower_bound,
        t1.variable_50p + 2.5 * (t3.variable_75p - t2.variable_25p) AS upper_bound
    FROM tb_50p AS t1
    LEFT JOIN tb_25p AS t2
    ON t1.group_id = t2.group_id
    LEFT JOIN tb_75p AS t3
    ON t1.group_id = t3.group_id
)
SELECT
    t1.*
FROM test_table AS t1
LEFT JOIN tb_bounds AS t2
ON t1.group_id = t2.group_id
WHERE t1.variable >= t2.lower_bound
AND t1.variable <= t2.upper_bound