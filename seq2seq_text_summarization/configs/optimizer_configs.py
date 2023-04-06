# We shall use the Adam or SGD optimizer.
# The configs are hyperparameters to be tuned.

OPTIMIZER = "Adam" # OR "SGD"
INITIAL_LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

ADAM_CONFIG = {
    "betas" : (0.9, 0.98),
}

SGD_CONFIG = {
    "momentum" : 0.9,
    "dampening" : 0.01,
}