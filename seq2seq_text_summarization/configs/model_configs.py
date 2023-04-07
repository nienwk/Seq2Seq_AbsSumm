# Configs for models

MODEL1_EMBEDDING_DIM = 512

MODEL1_ENCODER_HIDDEN_DIM = 512
MODEL1_ENCODER_NUM_LAYERS = 3
MODEL1_ENCODER_BIDIRECTIONAL = True # switch to False for submission

MODEL1_DECODER_HIDDEN_DIM = 512
MODEL1_DECODER_NUM_LAYERS = MODEL1_ENCODER_NUM_LAYERS # needs to match encoder and decoder as the hidden and cell states of LSTM needs to be matched

MODEL1_GLOBAL_DROPOUT_P = 0.1
MODEL1_ACTIVATION = "gelu"