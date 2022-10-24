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
    parser.add_argument('model_ckpt', help='model dir or file')
    parser.add_argument('input', help='input PGN')
    parser.add_argument('player', help='target player')
    parser.add_argument('--player_cat', type=int, help='')
    parser.add_argument('--player_set', type=str, help='')
    parser.add_argument('--game_set', help='')
    parser.add_argument('--num_processes', type=int, default = 5, help='processes in multiproc')
    parser.add_argument('--max_move', type=int, default = None)
    parser.add_argument('--min_move', type=int, default = None)
    parser.add_argument('--max_games', type=int, default = None)
    parser.add_argument('--table_name', type=str, default = 'raw_model_vectors')
    args = parser.parse_args()

    print(f"Loading  {args.input}", flush = True)

    global_infs = {
            'model_name': os.path.basename(args.model_ckpt).split('.')[0],
            'player_name': args.player,
            'player_category': args.player_cat,
            'game_set': args.game_set,
            'player_set' : args.player_set,
    }

    if args.min_move is not None:
        global_infs['min_move'] = args.min_move

    if args.max_move is not None:
        global_infs['max_move'] = args.max_move

    result_dicts = []
    model = analysis_helpers.Encoder_Model(args.model_ckpt)

    print(f"model loaded: {args.model_ckpt}", flush = True)

    with multiprocessing.Pool(args.num_processes) as pool:
        if args.max_games is None:
            game_vecs = pool.imap(make_game_array_wrap, ((g_dat, g_str, args.player, args.min_move, args.max_move) for g_dat, g_str in analysis_helpers.GamesFileStream(args.input)))
        else:
            games_set = get_n_games(args.input, args.max_games, args.min_move)
            game_vecs = pool.imap(make_game_array_wrap, ((g_dat, g_str, args.player, args.min_move, args.max_move) for g_dat, g_str in games_set))
        for i, game_ret in enumerate(game_vecs):
            try:
                if game_ret[1] is None:
                    continue
                else:
                    game_header, game_arr = game_ret
                    game_embed = model.embed_game(game_arr)
                    row_dict = global_infs.copy()
                    for cname in ['game_id', 'eco', 'white_elo', 'black_elo', 'time_control', 'white_won', 'black_won', 'no_winner', 'white_player', 'black_player']:
                        row_dict[cname] = analysis_helpers.per_game_infos[cname](game_header)
                    row_dict['timestamp'] = analysis_helpers.per_game_infos['datetime'](game_header)
                    row_dict['target_is_white'] = row_dict['white_player'] == args.player
                    row_dict['game_vec'] = game_embed.tolist()
                    result_dicts.append(row_dict)
                    print(i, end = ' ')
                    if i % 100 == 0:
                        print(flush = True)
            except:
                print('\n')
                print(game_ret)
                raise
    engine = sqlalchemy.create_engine(db_path)
    print()
    print(f"DB connected {engine}", flush = True)
    meta = sqlalchemy.MetaData()
    conn = engine.connect()
    results_table = sqlalchemy.Table(
                args.table_name,
                meta,
                autoload_with = conn,
                )
    conn.execute(results_table.insert(), result_dicts)

def get_n_games(input_path, n_target, min_move):
    if min_move is None:
        min_move = 0
    ret_sets = []
    games =  analysis_helpers.GamesFileStream(input_path)
    while len(ret_sets) < n_target:
        try:
            g_dat, g_str = next(games)
        except StopIteration:
            break
        num_moves = len(list(chess.pgn.read_game(io.StringIO(g_str)).mainline_moves())) // 2
        if num_moves - min_move < 10:
            continue
        else:
            ret_sets.append((g_dat, g_str))
    return ret_sets

def make_game_array_wrap(input_dat):
    try:
        return input_dat[0], analysis_helpers.make_game_array(input_dat[1],input_dat[2], min_move = input_dat[3], max_move = input_dat[4])
    except ValueError as e:
        if "unsupported variant" in str(e):
            return input_dat[0], None
        else:
            print(input_dat[2])
            print(input_dat[1])
            raise

if __name__ == "__main__":
    main()
