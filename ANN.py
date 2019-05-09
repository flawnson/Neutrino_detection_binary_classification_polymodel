# TODO: Change inputs and targets lists into tensor
#       Optimize hyperparameters

import torch
import lepton
import numpy as np

from torch import optim
from torch import nn

from torch.utils import data

import torch.nn.functional as F

"""IMPORT DATA"""

data = lepton.processed_datasets.pop()

X_data_train = lepton.X_data_pairs.pop()[0]
X_data_test = lepton.X_data_pairs.pop()[1]

Y_data_train = lepton.Y_data_pairs.pop()[0]
Y_data_test = lepton.Y_data_pairs.pop()[1]

train_loader = zip(X_data_train, Y_data_train)

"""MODEL"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(50, 64)
        self.drop_layer = nn.Dropout(.5)
        self.fc2 = nn.Linear(64, 128)
        self.drop_layer = nn.Dropout(.5)
        self.fc3 = nn.Linear(128, 32)
        self.drop_layer = nn.Dropout(.5)
        self.fc4 = nn.Linear(32, 2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

# create a complete CNN
model = Net()

# Loss function
criterion = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = 0.003, momentum= 0.9)

# number of epochs to train the model
n_epochs = 5 

valid_loss_min = np.Inf # track change in validation loss

train_on_gpu = False

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    # train the model #
    model.train()
    for data, target in train_loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
        
    # Validating the model
    model.eval()
    for data, target in valid_loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)
    
    # calculate average losses
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))