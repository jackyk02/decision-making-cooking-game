## Model parameters
residual_channels = 64
residual_blocks = 6
se_ratio = 8
seq_input_channels = 320 # input dimension to lstm/transformer
model_embedding_size = 512

## Training parameters
# learning_rate_init = 1e-4
learning_rate_init = 0.01
players_per_batch = 96
games_per_player = 24

v_players_per_batch = 96
v_games_per_player = 24
num_validate = 10
