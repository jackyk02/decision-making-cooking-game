from encoder.params_model import *
from collections import OrderedDict
from torch import nn
import torch
from encoder.transformer import ViT
import pandas as pd
import numpy as np
import seaborn as sns

sns.set(color_codes=True)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=True)),
        ]))


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, ratio):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lin1 = nn.Linear(channels, channels // ratio)
        self.relu = nn.ReLU(inplace=True)
        self.lin2 = nn.Linear(channels // ratio, 2 * channels)

    def forward(self, x):
        n, c, h, w = x.size()
        x_in = x

        x = self.pool(x).view(n, c)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)

        x = x.view(n, 2 * c, 1, 1)
        scale, shift = x.chunk(2, dim=1)

        x = scale.sigmoid() * x_in + shift
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels, se_ratio):
        super().__init__()
        # ResidualBlock can't be an nn.Sequential, because it would try to apply self.relu2
        # in the residual block even when not passed into the constructor
        self.layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(channels, channels, 3, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(channels)),
            ('relu', nn.ReLU(inplace=True)),

            ('conv2', nn.Conv2d(channels, channels, 3, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(channels)),

            ('se', SqueezeExcitation(channels, se_ratio)),
        ]))
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x_in = x

        x = self.layers(x)

        x = x + x_in
        x = self.relu2(x)
        return x


class Encoder(nn.Module):

    def __init__(self, loss_device):
        super().__init__()
        self.loss_device = loss_device

        channels = residual_channels

        self.conv_block = ConvBlock(3, channels, 3, padding=1)
        blocks = [(f'block{i + 1}', ResidualBlock(channels, se_ratio)) for i in range(residual_blocks)]
        self.residual_stack = nn.Sequential(OrderedDict(blocks))

        self.conv_block2 = ConvBlock(channels, channels, 3, padding=1)
        self.final_feature = ConvBlock(channels, seq_input_channels, 3, padding=1)
        self.global_avgpool = nn.AvgPool2d(kernel_size=2)

        self.cnn = nn.Sequential(*[
            self.conv_block,
            self.residual_stack,
            self.conv_block2,
            self.final_feature,
            self.global_avgpool,
            torch.nn.Flatten()
        ])

        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                m.weight.data.fill_(0.1)

        self.cnn.apply(init_weights)

        # Network defition
        self.transformer = ViT(input_dim=960,  # input dimension
                               output_dim=model_embedding_size,
                               dim=2,
                               depth=12,
                               heads=8,
                               mlp_dim=2048,
                               pool='mean',
                               dropout=0.,
                               emb_dropout=0.)

    def compute(self, games, max_tick):
        """
        Computes the embeddings for a batch of games in a round
        :param games: game features data frame for a round (with 170 unique ResponseIds)
        :return: the embeddings as a tensor of shape (number of games, embedding_size=512)
        """
        agg_game_vectors = torch.tensor([])

        for tick in max_tick:
            # for a single game
            # 2 * 3 * (2 * 26)
            # number of ticks for a player * 2 (workers) * (2 * 26)
            # row1 = [[games.iloc[0,:],games.iloc[0,:]], [games.iloc[0,:],games.iloc[0,:]], [games.iloc[0,:],games.iloc[0,:]]]
            # row2 = [[games.iloc[0,:],games.iloc[0,:]], [games.iloc[0,:],games.iloc[0,:]], [games.iloc[0,:],games.iloc[0,:]]]
            # rows = [row1, row2]
            game_features = []
            rows = []
            for i in range(tick):
                row = [[games.iloc[i * 3, :], games.iloc[i * 3, :]],
                       [games.iloc[i * 3 + 1, :], games.iloc[i * 3 + 1, :]],
                       [games.iloc[i * 3 + 2, :], games.iloc[i * 3 + 2, :]]]
                rows.append(row)
            torch_row = torch.tensor(rows)
            # print(torch_row.size())

            cnn_out = self.cnn(torch_row)
            game_features.append(cnn_out.unsqueeze(0))
            game_features = torch.cat(game_features, dim=0)
            # print(game_features.size())

            embeds_raw = self.transformer(game_features)
            embeds = embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)
            # print(embeds.size())
            agg_game_vectors = torch.cat((agg_game_vectors, embeds), dim=0)

        return agg_game_vectors


def computeEmbedding(round):
    # input is round no. and output is a list of game vectors in that round
    df = pd.read_csv("data.csv")
    df = df[df['round'] == round]
    df = df[np.invert(df['action'].isnull())]

    # max tick for a player
    max_tick = df.groupby('ResponseId').agg(max).reset_index()['tick']
    treatment = df['treatment']
    df = df.drop(columns=['treatment', 'round', 'worker', 'oid', 'ResponseId', 'completed'])

    nt_list = ['nt0_0', 'nt0_1', 'nt0_2', 'nt0_3', 'nt1_0', 'nt1_1', 'nt1_2', 'nt1_3', 'nt2_0', 'nt2_1', 'nt2_2',
               'nt2_3']
    df['nt_sum'] = df[nt_list].sum(axis=1)
    df = df.drop(columns=nt_list)

    ct_list = ['t0_0', 't0_1', 't0_2', 't1_0', 't1_1', 't1_2', 't2_0', 't2_1', 't2_2', 't0_3', 't1_3', 't2_3']
    df['t_sum'] = df[ct_list].sum(axis=1)
    df = df.drop(columns=ct_list)

    comp_list = ['completed_0', 'completed_1', 'completed_2', 'completed_3']
    df['completed_sum'] = df[comp_list].sum(axis=1)
    df = df.drop(columns=comp_list)

    nt_num_list = ['nt0', 'nt1', 'nt2']
    df['nt_sum2'] = df[nt_num_list].sum(axis=1)
    df = df.drop(columns=nt_num_list)

    task_list = ['t0', 't1', 't2']
    df['t_sum'] = df[task_list].sum(axis=1)
    df = df.drop(columns=task_list)

    worker_list = ['worker_0', 'worker_1', 'worker_2', 'worker_3']
    df['worker_sum'] = df[worker_list].sum(axis=1)
    df = df.drop(columns=worker_list)

    df = df.astype(np.float32)
    df.head(10)

    torch.manual_seed(100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Encoder(device)

    game_vectors_round = model.compute(df, max_tick)
    return game_vectors_round


if __name__ == "__main__":
    round = 1
    feature = computeEmbedding(round)
    print(feature)
    # feature = torch.round(feature, decimals=7)
    #round6 = computeEmbedding(4)
    #vectors = torch.cat((round5, round6), dim=0)
    #print(vectors.size())
    # t = feature.detach().numpy()
    # tf = pd.DataFrame(t)
    # tf.to_csv('feature_round3.csv', index = False)
    torch.save(feature, 'round' +  str(round) +'.pt')