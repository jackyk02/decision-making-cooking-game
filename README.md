# Player Embedding

## Environment
To set up the environment:
```
conda env create -f environment.yml
conda activate embed
```

## Data

Our player selection was based on private communications with the authors of [arxiv.org/abs/2008.10086](https://arxiv.org/abs/2008.10086). We are happy to share the data with other researchers privately, but for the privacy of the Lichess users we do not plan to make it public. Since the data are based on the lichess [database](https://database.lichess.org/) reproducing with a similar set of players is also possible.

Preprocess the data in the correct format (.npy.gz) from raw pgn file.
```
python preprocess.py --input_dir=INPUT_DIR
                     --output_dir=OUTPUT_DIR
```

## Training
Rearrange data folders so that they will be of structure `train/category/player_name/*.npy.gz`, where * is files of games with their game id as names. Structure of validation data is same as train.
```
python train.py RUN_ID
                --data_dir=TRAIN_ROOT_DIR
                --validate_data_dir=VALIDATE_ROOT_DIR
                --models_dir=MODEL_DIR
```
Other options such as validation frequency, model saving frequency, and whether to use visdom (visualization tool), please check [train.py](train.py). For other training and model hyparameters, please check out [params_data.py](encoder/params_data.py) and [params_model.py](encoder/params_model.py) for more details.

Note: Training will be done on multiple gpus by default.

## Analysis

Our data were written to a postgres database (`embed`) so our analysis code relies on it for recording results and caculating centriods. The sql files in `analysis/sql_tables` are required for `log_game_vecs.py` and `make_baseline_vec.py` which are run on the games. Then the analysis in `analysis/sql_tables/generate_summaries.sh` created the centriods for each of a our model and starting move sets in its own table. `log_game_vecs.py` and `make_baseline_vec.py` are run on pgn files containing only games by a single player and create the game vectors for each game encountered. The final stylometry accuracy uses the `get_dists_ranks()` function from `analysis_helpers/analysis` to compute the accuracy on sets of players.
