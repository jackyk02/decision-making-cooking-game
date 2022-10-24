import argparse
import enum
import os
import os.path
import chess.pgn
import io
import multiprocessing

import numpy as np

import torch
import sqlalchemy

os.environ['MAIA_DISABLE_TF'] = 'true'
import analysis_helpers

db_path = "postgresql:///embed"


def main():
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='input PGN')
    parser.add_argument('player', help='target player')
    parser.add_argument('--player_cat', type=int, help='')
    parser.add_argument('--player_set', type=str, help='')
    parser.add_argument('--game_set', help='')
    parser.add_argument('--max_games', type=int, default = None)
    parser.add_argument('--start_move', type=int, default = 0)
    parser.add_argument('--table_name', type=str, default = 'raw_baseline_vectors')
    args = parser.parse_args()

    name = f"baseline_{args.start_move + 5}"

    global_infs = {
            'model_name': name,
            'player_name': args.player,
            'player_category': args.player_cat,
            'game_set': args.game_set,
            'player_set' : args.player_set,
    }
    game_vecs = []
    result_dicts = []
    games = analysis_helpers.GamesFileStream(args.input)
    for g_dat, g_str in games:
        game_vecs.append((
            g_dat,
            analysis_helpers.make_vect(g_str, True if args.player == g_dat['White'] else False, min_move=args.start_move)
        ))
        if args.max_games is not None and len(game_vecs) >= args.max_games:
            break
    for i, (game_header, game_embed) in enumerate(game_vecs):
        try:
            row_dict = global_infs.copy()
            for cname in ['game_id', 'eco', 'white_elo', 'black_elo', 'time_control', 'white_won', 'black_won', 'no_winner', 'white_player', 'black_player']:
                row_dict[cname] = analysis_helpers.per_game_infos[cname](game_header)
            row_dict['timestamp'] = analysis_helpers.per_game_infos['datetime'](game_header)
            row_dict['target_is_white'] = row_dict['white_player'] == args.player
            row_dict['game_vec'] = analysis_helpers.make_vector(game_embed).tolist()
            result_dicts.append(row_dict)
            #print(i, end = ' ')
            #if i % 100 == 0:
            #    print(flush = True)
        except:
            print('\n')
            print(game_header, game_embed)
            raise
    engine = sqlalchemy.create_engine(db_path)
    print(args.input, args.start_move, 'done')
    #print(f"DB connected {engine}", flush = True)
    meta = sqlalchemy.MetaData()
    conn = engine.connect()
    results_table = sqlalchemy.Table(
                args.table_name,
                meta,
                autoload_with = conn,
                )
    conn.execute(results_table.insert(), result_dicts)

if __name__ == "__main__":
    main()
