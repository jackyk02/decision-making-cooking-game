CREATE TABLE raw_model_vectors (
    model_name text NOT NULL,
    player_name text NOT NULL,
    player_category bigint NOT NULL,
    player_set text,
    game_set varchar(8) NOT NULL,
    game_id varchar(8) NOT NULL,
    timestamp timestamp,
    white_player text,
    black_player text,
    white_elo int,
    black_elo int,
    target_is_white boolean,
    eco varchar(3),
    time_control text,
    white_won boolean,
    black_won boolean,
    no_winner boolean,
    game_vec real[512],
    attention_vec real[],
    min_move int,
    max_move int
);

CREATE INDEX ON raw_model_vectors (model_name);
CREATE INDEX ON raw_model_vectors (player_name);
CREATE INDEX ON raw_model_vectors (game_id);
CREATE INDEX ON raw_model_vectors (player_category);

