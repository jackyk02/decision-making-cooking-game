import sys
sys.path.append("../..")
from utils.leela_board._leela_board import LeelaBoard
from utils.leela_board.pgn_parser_re import *
from pathlib import Path
import argparse
import numpy as np
import os
import bz2
import gzip
import multiprocessing
from functools import partial
import random

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--input_dir", type=Path, default="/gamesdata/embed_data/raw_pgns")
    parser.add_argument("-o", "--output_dir", type=Path, default="/gamesdata/embed_data_processed")

    args = parser.parse_args()
    return args

def make_input(board):
    
    features = board.lcz_features()
    features = features.astype(np.float16) # (34, 8, 8), 34=2x13+8
    # features = np.reshape(features, (34, 8*8)) # (34, 64)

    return features

def parse_games(path, player, player_dir, save=True):
    # [[game1], [game2], [game3]...]
    games = []
    game = []
    game_id = ""
    num_positions = 0
    num_games = 0
    total_num_games = 0
    skip = False
    broken_game_id = ""
    num_move_threshold = 10

    entries = GamesFileStream(str(path)).iter_moves()
    for i, entry in enumerate(entries):
        game_type = entry['game_type']
        cur_game_id = entry['game_id']
        white_player = entry['white_player']
        active_player = entry['active_player']
        active_player = active_player.replace(',', '').replace(' ', '_')
        board_fen = entry['board']
        move = entry['move']
        num_ply = entry['num_ply']

        # skip broken game
        if cur_game_id == broken_game_id:
            continue

        # end of current game
        if cur_game_id != game_id:
            total_num_games += 1
            if len(game) != 0 and len(game) >= num_move_threshold:
                # stack list of boards into numpy array
                game = np.stack(game, axis=0)
                if save:
                    # https://stackoverflow.com/questions/42849821/how-to-recover-a-numpy-array-from-npy-gz-file?noredirect=1&lq=1
                    f_name = str(player_dir / game_id) + '.npy.gz'
                    f = gzip.GzipFile(f_name, "w")
                    np.save(f, game)
                    f.close()

                # games.append(game)
                num_games += 1

            # set skip flag if new game has fewer than threshold moves
            if os.path.exists(str(player_dir / cur_game_id) + '.npy.gz'):
                skip = True
                # print("{} exists, skip".format(str(player_dir / cur_game_id) + '.npy.gz'))
            else:
                if int(num_ply) // 2 < num_move_threshold:
                    skip = True
                else:
                    skip = False

            # reset to new game
            game = []
            game_id = cur_game_id

            if (board_fen != 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'):
                print("==================\n{}\n{}\n{}\n============".format(board_fen, player, game_id))
                raise
            # assert(board_fen == 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
            board = LeelaBoard(fen=board_fen)

        # only push moves with game > threshold moves
        if not skip:
            # push move instead of directly using fen, so we can have history information
            try:
                board.push_uci(move)
            except Exception as e:
                print(e)
                print(entry)
                print(move)
                print(board_fen)
                # set broken game id to skip
                broken_game_id = game_id
                # reset game
                game = []
                game_id = ""

            # skip entire game if broken game
            if game != "":
                # skip board positions if < 10 pieces
                if board.num_pieces() < 10:
                    skip = True
                elif player.lower() == active_player.lower():
                    try:
                        game.append(make_input(board))
                    except Exception as e:
                        print("{}\n{}\n{}\n".format(e, player, game_id))
                        exit(0)
                    num_positions += 1

    if not os.path.exists(str(player_dir / game_id) + '.npy.gz'):
        # last game
        if len(game) != 0 and len(game) >= num_move_threshold:
            # stack list of boards into numpy array
            game = np.stack(game, axis=0)
            if save:
                f_name = str(player_dir / game_id) + '.npy.gz'
                f = gzip.GzipFile(f_name, "w")
                np.save(f, game)
                f.close()

            num_games += 1
    # else:
    #     print("{} exists, skip".format(str(player_dir / game_id) + '.npy.gz'))

    # write info file
    f = open(str(player_dir / "info") + ".txt", "w")
    f.write("{}, {}, {}, {}\n".format(player, total_num_games, num_games, num_positions))
    f.close()
    return total_num_games, num_games, num_positions

# multi processing version
def multi_parse(output_dir, data_type, player_data):
    player = player_data.name[:-(9+len(data_type))]
    output_data_dir = output_dir / data_type / player
    output_data_dir.mkdir(exist_ok=True, parents=True)

    if (output_data_dir / 'info.txt').exists():
        print("player {} already complete".format(player))
        return

    num_valid_games = 0
    num_positions = 0
    total_num_games = 0
    total_num_games, num_valid_games, num_positions = parse_games(player_data, player, output_data_dir)

    print("{}: {}, {} games, {} valid games, {} board positions\n".format(player, data_type, total_num_games, num_valid_games, num_positions))

def process_player_pgn(input_dir, output_dir):
    # for num_games in ['0', '10', '1000', '5000', '10000', '20000', '30000', '40000']:
    for num_games in ['0', '10']:
        for data_train_type in ['explore', 'validate', 'holdout']:
            input_data_dir = input_dir / num_games / data_train_type
            output_data_dir = output_dir / num_games / data_train_type
            for data_type in ['train', 'validate', 'test']:
                player_data = list(input_data_dir.glob("*{}.pgn.bz2".format(data_type)))
                pool = multiprocessing.Pool(4)
                func = partial(multi_parse, output_data_dir, data_type)
                pool.map(func, player_data)
                pool.close()
                pool.join()

def process_player_pgn2(input_dir, output_dir):
    input_data_dir = input_dir
    player_data = input_data_dir.glob("lol.pgn.bz2")
    parse_games(player_data, 'lol', output_dir)

if __name__ == "__main__":
    args = parse_args()
    print(args.input_dir)
    players = process_player_pgn2(args.input_dir, args.output_dir)
