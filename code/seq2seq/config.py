from common_config import *

# training
batch_size = 5
lr_method = "sgd"  # sgd | adam | adagrad | rmsprop | momentum
lr = 1.0
lr_decay = 1.
lr_decay_strategy = None  # "on-no-improvement" | "step" | "exponential" | None

nepoch_no_imprv = 50
max_epochs = 400

# decoder
dim_tag = 150
tag_embeddings_dropout = 0.5
decoder_maximum_iterations = 12

# evaluation
eval_batch_size = 200
