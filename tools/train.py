import os
import sys

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from torch import nn, optim

from utils.train import TrainingLoop
 
# Define model and params
model = nn.Linear(768, 5)
csv_path = "data/normalized.csv"
batch_size = 16
loss_fn = nn.CrossEntropyLoss()  
optim_fn = optim.Adam

# Create train task
train_task = TrainingLoop(model, csv_path, batch_size, loss_fn, optim_fn, 0.001)

# Start train
train_task.train(n_epochs=1,save_name="test", eval_interval=1)