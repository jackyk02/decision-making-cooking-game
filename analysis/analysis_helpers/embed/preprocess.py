
import re
import numpy as np

from ..pgn_handling import GamesFileStream, get_game_info
from .leela_board import Embed_LeelaBoard

def make_input(board):

    features = board.lcz_features()
    features = features.astype(np.float16) # (34, 8, 8), 34=2x13+8

    return features

def parse_games(path, player):
    # [[game1], [game2], [game3]...]
    games = []
    game_ids = []
    game = []
    game_id = ""
    num_positions = 0
    num_games = 0
    total_num_games = 0
    skip = False
    broken_game_id = ""
    num_move_threshold = 10

    entries = GamesFileStream(path).iter_moves()
    for i, entry in enumerate(entries):
        game_type = entry['game_type']
        cur_game_id = entry['game_id']
        white_player = entry['white_player']
        active_player = entry['active_player']
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
                games.append(game)
                game_ids.append(cur_game_id)
                num_games += 1

            # set skip flag if new game has fewer than threshold moves
            if int(num_ply) // 2 < num_move_threshold:
                skip = True
            else:
                skip = False

            # reset to new game
            game = []
            game_id = cur_game_id

            assert(board_fen == 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
            board = Embed_LeelaBoard(fen=board_fen)

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
                elif player == active_player:
                    game.append(make_input(board))
                    num_positions += 1

    # last game
    if len(game) != 0 and len(game) >= num_move_threshold:
        # stack list of boards into numpy array
        game = np.stack(game, axis=0)
        games.append(game)
        game_ids.append(cur_game_id)
        num_games += 1

    return games, game_ids, total_num_games, num_games, num_positions

def preprocess_pgn(pgn_path, player):
    games, game_ids, total_num_games, num_valid_games, num_positions = parse_games(pgn_path, player)
    print("{}: {} games, {} valid games, {} board positions\n".format(player, total_num_games, num_valid_games, num_positions))
    return games, game_ids

def make_game_array(game_str, player, min_move = None, max_move = None):
    game = []
    if min_move is not None and max_move is not None:
        raise NotImplementedError("both min_move and max_move not supported")
    lines = list(get_game_info(game_str, no_clock = True))
    if len(lines) < 20:
        return None
    board_fen = lines[0]['board']
    board = Embed_LeelaBoard(fen=board_fen)
    move = lines[0]['move']
    active_player = lines[0]['active_player']
    for l in lines[1:]:
        board.push_uci(move)
        if player == active_player:
            game.append(make_input(board))
        if board.num_pieces() < 10:
            break
        move = l['move']
        active_player = l['active_player']
        if max_move is not None and len(game) >= max_move:
            break
    if min_move is not None:
        game = game[min_move:]
    if len(game) > 0:
        return np.stack(game, axis=0)
    else:
        return None
