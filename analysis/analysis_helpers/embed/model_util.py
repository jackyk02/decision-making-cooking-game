
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .params import *
from .model import Encoder
from .transformer import Attention

def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

class Recorder(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit

        self.data = None
        self.recordings = []
        self.hooks = []
        self.hook_registered = False
        self.ejected = False

    def _hook(self, _, input, output):
        self.recordings.append(output.clone().detach())

    def _register_hook(self):
        modules = find_modules(self.vit.transformer, Attention)
        for module in modules:
            handle = module.attend.register_forward_hook(self._hook)
            self.hooks.append(handle)
        self.hook_registered = True

    def eject(self):
        self.ejected = True
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.vit

    def clear(self):
        self.recordings.clear()

    def record(self, attn):
        recording = attn.clone().detach()
        self.recordings.append(recording)

    def forward(self, img):
        assert not self.ejected, 'recorder has been ejected, cannot be used anymore'
        self.clear()

        if not self.hook_registered:
            self._register_hook()

        pred = self.vit(img)
        attns = torch.stack(self.recordings, dim = 1)
        return pred, attns

class Encoder_Model:
    def __init__(self, weights_fpath):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(Path(weights_fpath))
        self._model_attn_wrapper = Recorder(self.model)

    def load_model(self, weights_fpath):
        checkpoint = torch.load(weights_fpath, self.device)
        model = Encoder(self.device)

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
        model.load_state_dict(new_state_dict)
        model = model.to(self.device)
        model.eval()

        print("Loaded encoder \"%s\" trained to step %d" % (weights_fpath.name, checkpoint["step"]))

        return model

    def compute_partial_slices(self, n_samples, partial_game_n_frames,
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

    def embed_frames_batch(self, frames_batch):
        """
        :param frames_batch: shape (batch_size, n_frames, 34, 8, 8)
        :return: the embeddings as a numpy array of float32 of shape (batch_size, model_embedding_size)
        """
        if self.model is None:
            raise Exception("Model was not loaded. Call load_model() before inference.")

        frames = torch.from_numpy(frames_batch).float().to(self.device)
        with torch.no_grad():
            embed = self.model.forward(frames).detach().cpu().numpy()

        return embed


    def embed_game(self, game, using_partials=True, return_partials=False):
        """
        Computes an embedding for a single game.
        """
        game = game[game_start:]    # ignore first 5 moves

        # Process the entire game if not using partials, or if number of positions are less than partials
        if not using_partials or len(game) <= partials_n_frames:
            # make batch size none
            embed = self.embed_frames_batch(game[None, ...])[0]
            if return_partials:
                return embed, None, None
            return embed

        # Compute where to split the game into partials and pad if necessary
        game_slices = self.compute_partial_slices(n_samples=len(game), partial_game_n_frames=partials_n_frames)
        max_game_length = game_slices[-1].stop
        if max_game_length >= len(game):
            game = np.pad(game, [(0, max_game_length - len(game)), (0, 0), (0, 0), (0, 0)], "constant")

        # Split the game into partials
        frames_batch = np.array([game[s] for s in game_slices])
        partial_embeds = self.embed_frames_batch(frames_batch)

        # Compute the game embedding from the partial embeddings
        raw_embed = np.mean(partial_embeds, axis=0)
        embed = raw_embed / np.linalg.norm(raw_embed, 2)

        if return_partials:
            return embed, partial_embeds, game_slices

        return embed


    def embed_player(self, games:list):
        raw_embed = np.mean([self.embed_game(game, return_partials=False) for game in games], axis=0)
        return raw_embed / np.linalg.norm(raw_embed, 2)

    def embed_games(self, games:list, num_test_games=4, drop_last=False):
        game_chunks = [games[x:x+num_test_games] for x in range(0, len(games), num_test_games)]
        if len(games) > num_test_games and len(games) % num_test_games != 0 and drop_last:
            game_chunks = game_chunks[:-1]

        return np.array([self.embed_player(chunk) for chunk in game_chunks])


    def embed_frames_batch_attn(self, frames_batch):
        if self._model_attn_wrapper is None:
            raise Exception("Model attention wrapper was not loaded. Call load_model() before inference.")

        frames = torch.from_numpy(frames_batch).float().to(self.device)
        with torch.no_grad():
            embed, attns = self._model_attn_wrapper.forward(frames)
            embed = embed.detach().cpu().numpy()
            attns = attns.detach().cpu().numpy()

        return embed, attns


    def embed_game_attn(self, game):
        game = game[game_start:]    # ignore first # moves
        embed, attns = self.embed_frames_batch_attn(game[None, ...]) # batch size 1

        return attns
