
set -e

max_procs=1

function gen_summary() {
    source_table=$1
    collection_name=$2
    where_section=$3

    filter_table="filtered_new_${collection_name}"
    collection_table="collection_new_${collection_name}"

    psql -d embed -c "DROP TABLE ${filter_table};" || true
    psql -d embed -c "DROP TABLE ${collection_table};" || true

    psql -d embed -c "CREATE TABLE ${filter_table} (LIKE ${source_table});"
    psql -d embed -c "ALTER TABLE ${filter_table} ADD COLUMN row_num int;"

    echo "$collection_name setup"

    psql -d embed -c "
    INSERT INTO ${filter_table}
    SELECT * FROM (
    SELECT *,
    ROW_NUMBER() OVER (PARTITION BY player_name, game_set) as row_num
    FROM ${source_table}
    WHERE ${where_section} AND player_category != 100
    ) as x
    WHERE row_num <= 100;"
    for index_name in "model_name" "player_name" "player_set" "player_category"; do
        psql -d embed -c "CREATE INDEX ON ${filter_table} (${index_name});"
    done

    psql -d embed -c "
    CREATE TABLE ${collection_table} (
    model_name text NOT NULL,
    player_name text NOT NULL,
    player_category bigint NOT NULL,
    player_set text,
    game_set text,
    eco_prefix_mode varchar(2),
    eco_mode varchar(3),
    games_used int,
    player_vec real[]
    );"
    for index_name in "model_name" "player_name" "player_set" "player_category" "game_set"; do
        psql -d embed -c "CREATE INDEX ON ${collection_table} (${index_name});"
    done
    for i in {100..1}; do
        echo $collection_name $i
        psql -d embed -c "INSERT INTO ${collection_table} (
            model_name,
            player_name,
            player_category,
            player_set,
            game_set,
            eco_prefix_mode,
            eco_mode,
            games_used,
            player_vec
        ) SELECT
            model_name,
            player_name,
            player_category,
            player_set,
            game_set,
            MODE() WITHIN GROUP ( ORDER BY SUBSTRING(eco,0,3)),
            MODE() WITHIN GROUP ( ORDER BY SUBSTRING(eco,0,4)),
            ${i},
            vec_sum(game_vec)
        FROM ${filter_table}
        WHERE row_num <= ${i}
        GROUP BY
            model_name,
            player_name,
            player_category,
            player_set,
            game_set;" && echo "done ${i} ${collection_name}" &

        while [ `echo $(pgrep -c -P$$)` -gt $max_procs ]; do
                #printf "waiting\r"
                sleep 1
        done

    done
    wait
    echo "Done $collection_name"
}

gen_summary "raw_model_vectors_min" "final_42000_min_15" "model_name = 'final_42000' AND min_move = 15"

gen_summary "raw_model_vectors" "final_42000" "model_name = 'final_42000' AND min_move IS NUll AND max_move IS NUll AND max_move IS NUll"

gen_summary "raw_model_vectors" "baseline_20500" "model_name = 'final_baseline_20500' AND min_move IS NUll AND max_move IS NUll"

gen_summary "raw_model_vectors" "top_player_31500" "model_name =  'final_top_player_31500' AND min_move IS NUll AND max_move IS NUll" &

gen_summary "raw_baseline_vectors" "baseline_5" "model_name = 'baseline_5'"

gen_summary "raw_baseline_vectors" "baseline_20" "model_name = 'baseline_20'"

gen_summary "raw_model_vectors_min" "top_player_31500_min_15" "model_name = 'final_top_player_31500' AND min_move = 15" &

gen_summary "raw_model_vectors_min" "baseline_20500_min_15" "model_name = 'final_baseline_20500' AND min_move = 15" &

wait
echo "done all"
