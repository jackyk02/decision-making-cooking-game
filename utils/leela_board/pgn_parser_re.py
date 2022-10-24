import re
import bz2
import io
import collections.abc

from .pgn_parsing_funcs import per_game_infos, per_move_funcs,per_move_time_funcs, per_move_eval_funcs

import chess.pgn

_header=r'''\[([A-Za-z0-9_]+)\s+"([^"]*)"\]'''
header_re =re.compile(_header)
_headers=r'(' + _header + r'\s*\n)+\n'
_moves=r'[^\[]*(\*|1-0|0-1|1/2-1/2)\s*\n'
_move = r"""([NBKRQ]?[a-h]?[1-8]?[\-x]?[a-h][1-8](?:=?[nbrqkNBRQK])?|[PNBRQK]?@[a-h][1-8]|--|Z0|O-O(?:-O)?|0-0(?:-0)?)|(\{.*)|(;.*)|(\$[0-9]+)|(\()|(\))|(\*|1-0|0-1|1\/2-1\/2)|([\?!]{1,2})"""

game_counter_re = re.compile('\[Event "')

game_re=re.compile("(" + _headers +")(.*?)(\*|1-0|0-1|1\/2-1\/2)", re.MULTILINE | re.DOTALL)

class GamesFileRe(collections.abc.Iterator):
    def __init__(self, path):
        if path.endswith('bz2'):
            with bz2.open(path, 'rt') as f:
                self.file_string = f.read()
        else:
            with open(path, 'r') as f:
                self.file_string = f.read()
        self.path = path
        self._len = None
        self.re_iter =game_re.finditer(self.file_string)

    def __len__(self):
        if self._len is None:
            self._len = len(game_counter_re.findall(self.file_string))
        return self._len

    def __next__(self):
        r = next(self.re_iter)
        if r is None:
            import pdb; pdb.set_trace()
        header = header_re.findall(r.group(1))
        return { k : v for k, v in header}, r.group(0)

    def iter_moves(self):
        return games_move_iter(self)

class GamesFileStream(GamesFileRe):
    def __init__(self, path):
        if path.endswith('bz2'):
            self.file = bz2.open(path, 'rt')
        else:
            self.file = open(path, 'rt')
        self.path = path
        self._len = -1
        self.re_iter = stream_iter(self.file)#game_re.finditer(self.file_string)

    def __del__(self):
        try:
            self.file.close()
        except:
            pass

def stream_iter(file_handle):
    current_game=file_handle.readline()
    for line in file_handle:
        if line.startswith('[Event '):
            yield game_re.match(current_game.strip())
            current_game = ''
        current_game += line
    if len(current_game.strip()) > 0:
        yield game_re.match(current_game.strip())

def games_move_iter(game_stream):
    for game_dict, game_str in game_stream:
        lines = get_game_info(game_str)
        for l in lines:
            yield l

def get_header_info(header_dict):
    gameVals = {}
    for name, func in per_game_infos.items():
        try:
            gameVals[name] = func(header_dict)
        except KeyError:
            gameVals[name] = None
    return gameVals

def get_game_info(input_game):
    if isinstance(input_game, str):
        game = chess.pgn.read_game(io.StringIO(input_game))
    else:
        game = input_game

    gameVals = {}
    for name, func in per_game_infos.items():
        try:
            gameVals[name] = func(game.headers)
        except KeyError:
            gameVals[name] = None

    gameVals['num_ply'] = len(list(game.mainline()))

    moves_values = []
    for i, node in enumerate(game.mainline()):
        board = node.parent.board()
        node_dict = gameVals.copy()
        node_dict['move_ply'] = i
        for name, func in per_move_funcs.items():
            node_dict[name] = func(node, gameVals)
        if len(node.comment) > 0:
            if r'%clk' in node.comment:
                for name, func in per_move_time_funcs.items():
                    node_dict[name] = func(node, gameVals)
            # if r'%eval' in node.comment:
            #     for name, func in per_move_eval_funcs.items():
            #         node_dict[name] = func(node, gameVals)
        moves_values.append(node_dict)
    return moves_values
