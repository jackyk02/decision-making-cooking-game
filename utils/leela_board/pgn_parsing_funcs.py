from .pgn_to_csv import cp_to_winrate

import uuid
import re
import functools

time_regex = re.compile(r'\[%clk\s(\d+):(\d+):(\d+(\.\d+)?)\]', re.MULTILINE)
eval_regex = re.compile(r'\[%eval\s([0-9.+-]+)|(#(-)?[0-9]+)\]', re.MULTILINE)

low_time_threshold = 30

# Header

def gen_game_id(header):
    if 'Link' in header:
        return header['Link'].split('/')[-1]
    elif 'lichess' in header['Site']:
        return header['Site'].split('/')[-1]
    else:
        return str(uuid.uuid4())

def gen_game_type(header):
    if 'lichess' in header['Site']:
        return header['Event'].split(' tournament')[0].replace(' game', '').replace('Rated ', '')
    else:
        return None

def gen_url(header):
    if 'lichess' in header['Site']:
        return header['Site']
    else:
        return header['Link']

def safe_to_int(s):
    try:
        return int(s)
    except ValueError:
        return 0

per_game_infos = {
    'game_id' : gen_game_id,
    'game_type' : gen_game_type,
    'is_lichess' : lambda x : 'lichess' in x['Site'],
    'date' : lambda x : x['UTCDate'],
    'time' : lambda x : x['UTCTime'],
    'url' : gen_url,
    'result' : lambda x : x['Result'],
    'white_player' : lambda x : x['White'],
    'black_player' : lambda x : x['Black'],
    'white_title' : lambda x : x.get('WhiteTitle', ''),
    'black_title' : lambda x : x.get('BlackTitle', ''),
    'white_elo' : lambda x : safe_to_int(x['WhiteElo']),
    'black_elo' : lambda x : safe_to_int(x['BlackElo']),
    'eco' : lambda x : x['ECO'],
    'time_control' : lambda x : x['TimeControl'],
    'termination' : lambda x : x['Termination'],
    'white_won' : lambda x : x['Result'] == '1-0',
    'black_won' : lambda x : x['Result'] == '0-1',
    'no_winner' : lambda x : x['Result'] not in  ['1-0', '0-1'],
}

# Moves

## No comment

@functools.lru_cache(maxsize=128)
def white_active(node):
    return node.parent.board().fen().split(' ')[1] == 'w'

per_move_funcs = {
    'move' : lambda n, d : str(n.move),
    'board' : lambda n, d : n.parent.board().fen(),
    'white_active' : lambda n, d : white_active(n),
    'active_player' : lambda n, d : d['white_player'] if white_active(n) else d['black_player'],
    'is_capture' : lambda n,d : n.parent.board().is_capture(n.move),
    'active_won' : lambda n,d : d['white_won'] if white_active(n) else d['black_won'],
    'active_elo' : lambda n,d : d['white_elo'] if white_active(n) else d['black_elo'],
    'opponent_elo' : lambda n,d : d['white_elo'] if not white_active(n) else d['black_elo'],
}

# Clock

def get_move_clock(comment):
    timesRe = time_regex.search(comment)
    return  int(timesRe.group(1)) * 60 * 60 + int(timesRe.group(2)) * 60  + float(timesRe.group(3))

def get_opp_clck(node):
    pc = node.parent.comment
    if len(pc) > 0:
        return get_move_clock(pc)
    else:
        # Start of game
        return get_move_clock(node.comment)

def time_control_to_secs(timeStr, moves_per_game = 35):
    if timeStr == '-':
        return 10800 # 180 minutes per side max on lichess
    else:
        try:
            t_base, t_add = timeStr.split('+')
            return int(t_base) + int(t_add) * moves_per_game
        except ValueError:
            return int(timeStr)

per_move_time_funcs = {
    'clock' : lambda n, d : get_move_clock(n.comment),
    'clock_percent' : lambda n, d : get_move_clock(n.comment) / time_control_to_secs(d['time_control']),
    'opp_clock' : lambda n, d : get_opp_clck(n),
    'opp_clock_percent' : lambda n, d: get_opp_clck(n) / time_control_to_secs(d['time_control']),
    'low_time' : lambda n, d : get_move_clock(n.comment) < low_time_threshold,
}

# Eval
def get_move_eval(comment):
    if len(comment) < 1:
        return 0.1
    cp_str = eval_regex.search(comment).group(1)
    try:
        return float(cp_str)
    except TypeError:
        if '-' in comment:
            return float('-inf')
        else:
            return float('inf')

def get_cp_loss(node):
    is_white = white_active(node)
    cp_par = get_move_eval(node.parent.comment) * (1 if is_white else -1)
    cp_aft = get_move_eval(node.comment) * (1 if is_white else -1)
    return cp_par - cp_aft


def get_move_wr(comment):
    return cp_to_winrate(get_move_eval(comment))

per_move_eval_funcs = {
    'cp' : lambda n, d : get_move_eval(n.parent.comment),
    'cp_rel' : lambda n, d : get_move_eval(n.parent.comment) * (1 if white_active(n) else -1),
    'cp_loss' : lambda n, d : get_cp_loss(n),
    'winrate' : lambda n, d : get_move_wr(n.parent.comment) if white_active(n) else (1 - get_move_wr(n.parent.comment)),
    'opp_winrate' : lambda n, d : (1 - get_move_wr(n.parent.comment)) if white_active(n) else get_move_wr(n.parent.comment),
    'winrate_loss' : lambda n, d : (get_move_wr(n.parent.comment) - get_move_wr(n.comment)) if white_active(n) else (- get_move_wr(n.parent.comment) + get_move_wr(n.comment)),
}


full_options_lst = ['move_ply'] + list(per_game_infos) + list(per_move_funcs) + list(per_move_time_funcs) + list(per_move_eval_funcs)