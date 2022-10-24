import io
import os.path

import chess.pgn
import numpy as np

full_uci_lookup = {}
index = 0
for c1 in range(8):
    for n1 in range(8):
        for c2 in range(8):
            for n2 in range(8):
                full_uci_lookup[f"{chr(97 + n1)}{c1 + 1}{chr(97 + n2)}{c2 + 1}"] = index
                index += 1
full_uci_lookup_black = {}
index = 0
for c1 in range(8):
    for n1 in range(8):
        for c2 in range(8):
            for n2 in range(8):
                full_uci_lookup_black[f"{chr(97 + (7 - n1))}{c1 + 1}{chr(97 + (7 - n2))}{c2 + 1}"] = index
                index += 1

def make_vect(g_str, is_white, min_move = 0):
    g = chess.pgn.read_game(io.StringIO(g_str))
    moves = [m.uci() for m in g.mainline_moves()][min_move * 2:10 + min_move * 2]
    if is_white:
        return [full_uci_lookup.get(m,0) for m in moves[::2]]
    else:
        return [full_uci_lookup_black.get(m,0) for m in moves[1::2]]

def make_vector(c_arrays):
    counts = np.array(c_arrays).flatten()
    ret_vec = np.zeros((8*8*8*8,))
    for v in counts:
        ret_vec[v] += 1
    return ret_vec
