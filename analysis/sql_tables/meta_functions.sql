CREATE OR REPLACE FUNCTION vec_add(arr1 real[], arr2 real[])
RETURNS real[] AS
$$
SELECT array_agg(result)
FROM (SELECT tuple.val1 + tuple.val2 AS result
      FROM (SELECT UNNEST($1) AS val1
                   ,UNNEST($2) AS val2
                   ,generate_subscripts($1, 1) AS ix) tuple
      ORDER BY ix) inn;
$$ LANGUAGE SQL IMMUTABLE STRICT;

CREATE AGGREGATE vec_sum(real[]) (
    SFUNC = vec_add
    ,STYPE = real[]
);
