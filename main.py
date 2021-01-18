# This code is used to train a lstm nn 
import numpy as np
import librosa
import os
import csv
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import time

MODEL_DIR = os.path.join("Models" + os.sep)

training_data = np.load("audiodata.npy", allow_pickle = True)
tb = SummaryWriter()

# Hyperparameters

# The number of LSTM blocks per layer.
hidden_size = 512
# The number of input features per time-step.
input_size = 94
# The number of MFCCs 
sequence_length = 20
# The number of hidden layers.
num_layers = 1
# The number of classes (noise, speech, music)
num_classes = 3

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        
        super(Net, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.dropout = nn.Dropout(p = 0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
        
        
    def forward(self, x):
        # Initial states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # LSTM
        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = F.relu(x)
        # Linear layers
        x = self.dropout(x)
        x = self.fc(x)

        result = F.log_softmax(x, dim = 1)
        
        return result

def fwd_pass(X, y, train = False):
    # Zero the gradient when training
    # We need to do this because PyTorch accumulates the gradients on subsequent backward passes
    if train:
        net.zero_grad()
    # Pass the data through the network
    outputs = net(X)
    # Count accuracy and loss
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)
    loss = loss_function(outputs, y)
    # Backpropagation when training
    if train:
        loss.backward()
        optimizer.step()
    
    return acc, loss

def test(size = 32):
    # Random start makes sure we don't test on the same data
    random_start = np.random.randint(len(test_X - size))
    X, y = test_X[random_start:random_start + size], test_y[random_start: random_start + size]
    net.eval()
    # We don't update the gradients when testing
    with torch.no_grad():
        val_acc, val_loss = fwd_pass(X.view(-1, 20, 94).to(device), y.to(device))
    return val_acc, val_loss
    
# Initializing the nn object      
net = Net(input_size, hidden_size, num_layers, num_classes)
# Defining optimizer and loss function
optimizer = optim.Adam(net.parameters(), lr = 0.0002, weight_decay = 0.0000001)                         
loss_function = nn.MSELoss()

# Converting a NumPy array of MFCCs into a PyTorch Tensor
X = torch.Tensor([i[0] for i in training_data]).view(-1, 20, 94)
# Converting a NumPy array of labels into a PyTorch Tensor
y = torch.Tensor([i[1] for i in training_data])
# Converting the MFCC values from [0:255] to [0:1]
# Neural networks in general work better with the latter
X = X/255.0

# Specifies which percentage of data is used for testing
VAL_PCT = 0.1
val_size = int(len(X)*VAL_PCT)

# Splits the data into a training and testing sets
train_X = X[:-val_size]
train_y = y[:-val_size]
test_X = X[-val_size:]
test_y = y[-val_size:]

# Frees up memory
training_data = None
X = None
y = None
# Makes sure the nn is in training mode
net.train()

# Computing on a Nvidia CUDA card if the machine has one
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    net.to(device)


def train():
    BATCH_SIZE = 2048
    EPOCHS = 150
     # Model name is used to save models
    MODEL_NAME = f"model-{EPOCHS}-epochs-{BATCH_SIZE}-batch-{int(time.time())}"
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
            net.train()
            # Taking a batch of data
            batch_X = train_X[i:i + BATCH_SIZE].view(-1, 20, 94)
            batch_y = train_y[i:i + BATCH_SIZE]
            # Transfering it to a GPU if possible
            if torch.cuda.is_available():
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            # Passing the data through the network, storing accuracy and loss
            acc, loss = fwd_pass(batch_X, batch_y, train = True)
            
        # Testing the model on the test data
        val_acc, val_loss = test(3000)
        
        # Saving the model
        torch.save(net.state_dict(), MODEL_DIR + "lstm-audio" + MODEL_NAME + "accuracy-" + str(val_acc))
        # Monitoring the results with TensorBoard  
        tb.add_scalar('train_loss', loss, epoch)
        tb.add_scalar('train_acc', acc, epoch)
        tb.add_scalar('val_loss', val_loss, epoch)
        tb.add_scalar('val_acc', val_acc, epoch)
    

TRAIN_MODEL = True
if TRAIN_MODEL:
    train()
    print("TRAINING FINISHED")
