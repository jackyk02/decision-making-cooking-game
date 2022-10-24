from encoder.params_data import *
from encoder.params_model import *
from encoder.model import Encoder
from encoder.recorder import Recorder
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import numpy as np
import torch
import time
_model = None # type: Encoder
_resnet = None
_device = None # type: torch.device
_model_attn_wrapper = None

def load_model(weights_fpath: Path, multi_gpu=False, device=None, use_resnet=False):
    """
    Loads the model in memory. If this function is not explicitely called, it will be run on the 
    first call to embed_frames() with the default weights file.
    
    :param weights_fpath: the path to saved model weights.
    :param device: either a torch device or the name of a torch device (e.g. "cpu", "cuda"). The 
    model will be loaded and will run on this device. Outputs will however always be on the cpu. 
    If None, will default to your GPU if it"s available, otherwise your CPU.
    """
    # TODO: I think the slow loading of the encoder might have something to do with the device it
    #   was saved on. Worth investigating.
    global _model, _resnet, _device, _model_attn_wrapper

    if device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        _device = torch.device(device)
    else:
        _device = device

    checkpoint = torch.load(weights_fpath, _device)
    _model = Encoder(_device)

    if multi_gpu:
        if torch.cuda.device_count() <= 1:
            raise "multi_gpu cannot be enabled"

        _model = torch.nn.DataParallel(_model)
        # load params
        _model.load_state_dict(checkpoint["model_state"])
        _resnet = torch.nn.DataParallel(_model.module.cnn)
    else:
        # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
        state_dict = checkpoint['model_state']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module' in k:
                name = k[7:] # remove `module.` because of DataParallel
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v

        # load params
        _model.load_state_dict(new_state_dict)
        _resnet = _model.cnn

    if use_resnet:
        _resnet = _resnet.to(_device)
        _resnet.eval()
    else:
        _model = _model.to(_device)
        _model.eval()

    _model_attn_wrapper = Recorder(_model)
    # print("Loaded encoder \"%s\" trained to step %d" % (weights_fpath.name, checkpoint["step"]))
    
    
def is_loaded():
    return _model is not None


def compute_partial_slices(n_samples, partial_game_n_frames=partials_n_frames,
                           min_pad_coverage=0.8, overlap=0.5):
    """
    :param n_samples: the number of moves in the game
    :param partial_game_n_frames: the number of frames in each partial game
    :param min_pad_coverage: when reaching the last partial game, it may or may not have 
    enough frames. If at least <min_pad_coverage> of <partial_game_n_frames> are present, 
    then the last partial game will be considered, as if we padded the audio. Otherwise, 
    it will be discarded, as if we trimmed the audio. If there aren't enough frames for 1 partial 
    game, this parameter is ignored so that the function always returns at least 1 slice.
    :param overlap: by how much the partial game should overlap. If set to 0, the partial 
    games are entirely disjoint. 
    :return: partial games frame range.
    """
    assert 0 <= overlap < 1
    assert 0 < min_pad_coverage <= 1
    assert(n_samples > partial_game_n_frames)

    # Compute the slices
    frame_step = max(int(np.round(partial_game_n_frames * (1 - overlap))), 1)
    steps = int(np.ceil(n_samples / frame_step)) * frame_step

    game_slices = []
    for i in range(0, steps, frame_step):
        game_range = np.array([i, i + partial_game_n_frames])
        game_slices.append(slice(*game_range))
        
    # Evaluate whether extra padding is warranted or not
    last_game_range = game_slices[-1]
    coverage = (n_samples - last_game_range.start) / (last_game_range.stop - last_game_range.start)
    if coverage < min_pad_coverage and len(game_slices) > 1:
        game_slices = game_slices[:-1]
    
    return game_slices

def embed_frames_batch(frames_batch):
    """    
    :param frames_batch: shape (batch_size, n_frames, 34, 8, 8)
    :return: the embeddings as a numpy array of float32 of shape (batch_size, model_embedding_size)
    """
    if _model is None:
        raise Exception("Model was not loaded. Call load_model() before inference.")
    
    frames = torch.from_numpy(frames_batch).float().to(_device)
    with torch.no_grad():
        embed = _model.forward(frames).detach().cpu().numpy()

    return embed

def embed_game(game, using_partials=True, return_partials=False, **kwargs):
    """
    Computes an embedding for a single game.
    
    # TODO: handle multiple wavs to benefit from batching on GPU
    :param game: a preprocessed game as a numpy array of float32, (num_moves, 34, 8, 8)
    :param using_partials: if True, then the game is split in partial games of 
    <partial_game_n_frames> frames and the game embedding is computed from their 
    normalized average. If False, the game is instead computed from feeding the entire 
    game array to the network.
    :param return_partials: if True, the partial embeddings will also be returned along with the 
    wav slices that correspond to the partial embeddings.
    :param kwargs: additional arguments to compute_partial_splits()
    :return: the embedding as a numpy array of float32 of shape (model_embedding_size,). If 
    <return_partials> is True, the partial games as a numpy array of float32 of shape 
    (n_partials, model_embedding_size) and the game partials as a list of slices will also be 
    returned. If <using_partials> is simultaneously set to False, both these values will be None 
    instead.
    """

    game = game[game_start:]    # ignore first 5 moves

    # Process the entire game if not using partials, or if number of positions are less than partials
    if not using_partials or len(game) <= partials_n_frames:
        # make batch size none
        embed = embed_frames_batch(game[None, ...])[0]
        if return_partials:
            return embed, None, None
        return embed
    
    # Compute where to split the game into partials and pad if necessary
    game_slices = compute_partial_slices(len(game), **kwargs)
    max_game_length = game_slices[-1].stop
    if max_game_length >= len(game):
        game = np.pad(game, [(0, max_game_length - len(game)), (0, 0), (0, 0), (0, 0)], "constant")

    # Split the game into partials
    frames_batch = np.array([game[s] for s in game_slices])
    partial_embeds = embed_frames_batch(frames_batch)
    
    # Compute the game embedding from the partial embeddings
    raw_embed = np.mean(partial_embeds, axis=0)
    embed = raw_embed / np.linalg.norm(raw_embed, 2)
    
    if return_partials:
        return embed, partial_embeds, game_slices

    return embed

def embed_player(games:list, **kwargs):
    """
    Compute the embedding of a collection of games by averaging their embedding and L2-normalizing it.
    :param games: list of games a numpy arrays
    :param kwargs: extra arguments to embed_game()
    :return: the embedding as a numpy array of float32 of shape (model_embedding_size,).
    """
    raw_embed = np.mean([embed_game(game, return_partials=False, **kwargs) \
                            for game in games], axis=0)
    return raw_embed / np.linalg.norm(raw_embed, 2)

def embed_games(games:list, num_test_games=4, drop_last=False):
    game_chunks = [games[x:x+num_test_games] for x in range(0, len(games), num_test_games)]
    if len(games) > num_test_games and len(games) % num_test_games != 0 and drop_last:
        game_chunks = game_chunks[:-1]

    return np.array([embed_player(chunk) for chunk in game_chunks])

def plot_embedding_as_heatmap(embed, ax=None, title="", shape=None, color_range=(0, 0.30)):
    if ax is None:
        ax = plt.gca()
    
    if shape is None:
        height = int(np.sqrt(len(embed)))
        shape = (height, -1)
    embed = embed.reshape(shape)
    
    cmap = cm.get_cmap()
    mappable = ax.imshow(embed, cmap=cmap)
    cbar = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_clim(*color_range)

    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(title)

def embed_frames_batch_resnet(frames_batch):
    """    
    :param frames_batch: shape (batch_size, n_frames, 34, 8, 8)
    :return: the embeddings as a numpy array of float32 of shape (batch_size, model_embedding_size)
    """
    if _resnet is None:
        raise Exception("Model(resnet) was not loaded. Call load_model() before inference.")
    frames = torch.from_numpy(frames_batch).float().to(_device)

    batch_size, n_frames, feature_shape = frames.shape[0], frames.shape[1], frames.shape[2:]
    frames = torch.reshape(frames, (batch_size*n_frames, *feature_shape))

    with torch.no_grad():
        embed = _resnet.forward(frames).detach().cpu().numpy()

    # normalize the n_frames dimension
    embed = np.reshape(embed, (batch_size, n_frames, embed.shape[-1]))
    embed = np.mean(embed, axis=1)
    embed = embed / np.linalg.norm(embed, 2, axis=1, keepdims=True)
    return embed

def embed_player_resnet(games:list, **kwargs):
    """
    Compute the embedding of a collection of games by averaging their embedding and L2-normalizing it.
    :param games: list of games a numpy arrays
    :param kwargs: extra arguments to embed_game_resnet()
    :return: the embedding as a numpy array of float32 of shape (model_embedding_size,).
    """
    raw_embed = np.mean([embed_game_resnet(game, return_partials=False, **kwargs) \
                            for game in games], axis=0)
    return raw_embed / np.linalg.norm(raw_embed, 2)

def embed_games_resnet(games:list, num_test_games=4, drop_last=False):
    game_chunks = [games[x:x+num_test_games] for x in range(0, len(games), num_test_games)]
    if len(games) > num_test_games and len(games) % num_test_games != 0 and drop_last:
        game_chunks = game_chunks[:-1]

    return np.array([embed_player_resnet(chunk) for chunk in game_chunks])

def embed_game_resnet(game, using_partials=True, return_partials=False, **kwargs):
    """
    Computes an embedding for a single game.
    
    # TODO: handle multiple wavs to benefit from batching on GPU
    :param game: a preprocessed game as a numpy array of float32, (num_moves, 34, 8, 8)
    :param using_partials: if True, then the game is split in partial games of 
    <partial_game_n_frames> frames and the game embedding is computed from their 
    normalized average. If False, the game is instead computed from feeding the entire 
    game array to the network.
    :param return_partials: if True, the partial embeddings will also be returned along with the 
    wav slices that correspond to the partial embeddings.
    :param kwargs: additional arguments to compute_partial_splits()
    :return: the embedding as a numpy array of float32 of shape (model_embedding_size,). If 
    <return_partials> is True, the partial games as a numpy array of float32 of shape 
    (n_partials, model_embedding_size) and the game partials as a list of slices will also be 
    returned. If <using_partials> is simultaneously set to False, both these values will be None 
    instead.
    """

    game = game[game_start:]    # ignore first 5 moves

    # Process the entire game if not using partials, or if number of positions are less than partials
    if not using_partials or len(game) <= partials_n_frames:
        # make batch size none
        embed = embed_frames_batch_resnet(game[None, ...])[0]
        if return_partials:
            return embed, None, None
        return embed
    
    # Compute where to split the game into partials and pad if necessary
    game_slices = compute_partial_slices(len(game), **kwargs)
    max_game_length = game_slices[-1].stop
    if max_game_length >= len(game):
        game = np.pad(game, [(0, max_game_length - len(game)), (0, 0), (0, 0), (0, 0)], "constant")

    # Split the game into partials
    frames_batch = np.array([game[s] for s in game_slices])
    partial_embeds = embed_frames_batch_resnet(frames_batch)
    
    # Compute the game embedding from the partial embeddings
    raw_embed = np.mean(partial_embeds, axis=0)
    embed = raw_embed / np.linalg.norm(raw_embed, 2)
    
    if return_partials:
        return embed, partial_embeds, game_slices

    return embed

def embed_frames_batch_attn(frames_batch):
    if _model_attn_wrapper is None:
        raise Exception("Model attention wrapper was not loaded. Call load_model() before inference.")

    frames = torch.from_numpy(frames_batch).float().to(_device)
    with torch.no_grad():
        embed, attns = _model_attn_wrapper.forward(frames)
        embed = embed.detach().cpu().numpy()
        attns = attns.detach().cpu().numpy()

    return embed, attns


def embed_game_attn(game):
    game = game[game_start:]    # ignore first # moves
    embed, attns = embed_frames_batch_attn(game[None, ...]) # batch size 1
    
    return attns
