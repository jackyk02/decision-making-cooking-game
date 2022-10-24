`GE2E/encoder` has all the main code for models, data loader, etc.

`inference.py`: helper code to load trained model and run on game(s)

`model.py`: contains implementations of residual blocks and whole network definition plus the GE2E loss code

`params_data.py`: parameters for where the game starts (how many opening moves are discarded), and how many moves in a sequence to feed into the transformer

`params_model.py`: parameters for network architecture and training parameters such as learning rate and batch size

`recorder.py`: helper code to retrieve attention maps, see inference.py for usage 

`train.py`: main code for training

`transformer.py`: transformer code, which is used in model.py

`visualizations.py`: visdom server code for training stats visualizations

train/val/test data location: /gamesdata/embed_data_processed on maia-compute. It has a README to explain each folder/dataset
