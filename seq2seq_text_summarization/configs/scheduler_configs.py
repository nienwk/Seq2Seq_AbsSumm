# For use with SequentialLR
MILESTONE_RATIO = 0.2
MILESTONE_HARD = 1000 # number of iterations

LINEAR_LR_CONFIG = {
    "start_factor" : 0.5,
    # To be set based on batch size, and choice of stepping by batch or stepping by epoch
    # "total_iters" : min(int(MILESTONES_RATIO * len(dataloader) // 1) , MILESTONE_HARD) # by iteration
    # OR
    # "total_iters" : min(int(MILESTONES_RATIO * NUM_EPOCHS // 1) , MILESTONE_HARD) # by epoch
}

COSINE_ANNEALING_CONFIG = {
    # To be set based on batch size, and choice of stepping by batch or stepping by epoch
    # "T_max" : len(dataloader) - LINEAR_LR_CONFIG["total_iters"] # by iteration
    # OR
    # "T_max" : NUM_EPOCHS - LINEAR_LR_CONFIG["total_iters"] # by epoch
    "eta_min" : 1e-6,
}