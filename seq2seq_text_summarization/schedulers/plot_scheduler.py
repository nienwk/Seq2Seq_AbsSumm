import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import torch.nn as nn
from ..configs.scheduler_configs import *
from ..configs.optimizer_configs import INITIAL_LEARNING_RATE

# This script is meant to be executed standalone to visualize the learning rate schedule
# Also used to look at the structure of how to setup cosine annealing learning rate with linear warmup

initial_lr = INITIAL_LEARNING_RATE
layer = nn.Linear(10,10)
max_iteration = 6839 # (= 54712 number of training samples / 8 batch size)
warmup = min(int((MILESTONE_RATIO * max_iteration) // 1.0), MILESTONE_HARD)
print(f"warmup = {warmup}")
optimizer = optim.SGD(layer.parameters(), initial_lr)
scheduler = optim.lr_scheduler.SequentialLR(optimizer,\
    [ LinearLR(optimizer, LINEAR_LR_CONFIG['start_factor'], total_iters=warmup), CosineAnnealingLR(optimizer, max_iteration-warmup, COSINE_ANNEALING_CONFIG['eta_min']) ], [warmup])

lr = []
for i in range(max_iteration):
    optimizer.step()
    scheduler.step()
    lr.append(scheduler.get_last_lr())

print(len(lr))
print((lr[warmup-2], lr[warmup-1], lr[warmup], lr[warmup+1], lr[max_iteration-1]))
plt.plot([i for i in range(1,len(lr)+1)], lr)
plt.savefig("scheduler_plot.png")
plt.close()
