CREATE TABLE meta_player_games (
    player text NOT NULL,
    player_set text NOT NULL,
    player_rating int NOT NULL,
    game_set text NOT NULL,
    game_id varchar(8) NOT NULL,

    opponent text NOT NULL,
    is_white boolean NOT NULL,
    player_rating_game int NOT NULL,
    opponent_rating_game int NOT NULL,
    primary key (player, game_id)
);

CREATE INDEX idx_meta_player_gamesplayer_set ON meta_player_games (player_set);
CREATE INDEX idx_meta_player_gamesplayer_rating ON meta_player_games (player_rating);
CREATE INDEX idx_meta_player_games_game_set ON meta_player_games (game_set);


--SELECT player_set, player_rating, COUNT(*) FROM (SELECT player, player_set, player_rating FROM meta_player_games GROUP BY player, player_set, player_rating) as x GROUP BY player_set, player_rating ORDER BY player_set, player_rating;

--SELECT player, rating, count FROM (SELECT player, AVG(player_rating)::INT as rating, COUNT(*) as count FROM meta_player_games GROUP BY player) as x WHERE count < rating ORDER BY rating, COUNT;
