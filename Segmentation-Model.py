#Importing the modules we need
import torch
import spconv
import sparseconvnet as scn
import torch.utils.data as data_utils
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import argparse
import multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import time
import math
from statistics import mean
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
 
#Set directory to store log files for graphing
writer = SummaryWriter('logs/fit/Sparse-Segmentation')
 
#Load data and convert into PyTorch datasets for training and testing
X = np.load('train_X64.npy')
y = np.load('train_y64.npy')
X, X_val, y, y_val = train_test_split(X, y, test_size=1/3, random_state=43)
X_train = torch.from_numpy(X).float()
y_train = torch.from_numpy(y).long()
X_val = torch.from_numpy(X_val).float()
y_val = torch.from_numpy(y_val).long()
train_data = data_utils.TensorDataset(X_train, y_train)
val_data = data_utils.TensorDataset(X_val, y_val)
 
#Create and load data into data generators
train_loader = torch.utils.data.DataLoader(train_data,
       batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(val_data,
       batch_size=128, shuffle=True)
#Define model
class Net(nn.Module):
   def __init__(self):
       super(Net, self).__init__()
       self.net = spconv.SparseSequential(
           spconv.SparseConv2d(1, 64, 3, 1, indice_key="cp0"),
           spconv.SparseInverseConv2d(64, 32, 3, indice_key="cp0"), # need provide kernel size to create weight
       )
 
   def forward(self, x: torch.Tensor):
       x_sp = spconv.SparseConvTensor.from_dense(x.reshape(-1, 64, 64, 1))
       return self.net(x_sp).dense()
 
model = Net().to(device)
 
#Create Jaccard score metric
def jac(y_true, y_pred, labels):
   return jaccard_score(y_true, y_pred, labels, average='weighted')
 
#Create function for testing/validating the model
def test(model, device, test_loader, epoch):
   model.eval()
   correct = 0
   total = 0
   jaccard = []
   with torch.no_grad():
       for data, target in test_loader:
           data, target = data.to(device), target.to(device)
           target = target.squeeze(3)
           output = model(data)
           _, predicted = torch.max(output.data, 1)
           total += target.nelement()
           correct += predicted.eq(target.data).sum().item()
           trgt = target.cpu().numpy().reshape(-1)
           pred = predicted.cpu().numpy().reshape(-1)
           jaccard.append(jac(trgt, pred, [0, 1, 2]))
   jac_score = mean(jaccard)
   test_acc = 100 * correct / total
   print('Test Epoch: {} Acc: {} Jaccard Score: {}'.format(epoch, test_acc, jac_score))
   writer.add_scalar('val_accuracy', correct/total, epoch)
   writer.add_scalar('val_jaccard', jac_score, epoch)
 
#Train model
num_epochs = 50
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
for epoch in range(1, num_epochs + 1):
   model.train()
   acc = 0
   correct = 0
   total = 0
   jac_scoreList = []
   for batch_idx, (data, mask) in enumerate(train_loader):
       data, mask = data.to(device), mask.to(device)
       mask = mask.squeeze(3)
       output = model(data)
       softmax = F.log_softmax(output, dim=1)
       loss = criterion(softmax, mask)
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
      
       _, predicted = torch.max(output.data, 1)
       total += mask.nelement()
       correct += predicted.eq(mask.data).sum().item()
       acc = 100 * correct / total
       trgt = mask.cpu().numpy().reshape(-1)
       pred = predicted.cpu().numpy().reshape(-1)
       jac_scoreList.append(jac(trgt, pred, [0, 1, 2]))
      
       if batch_idx % 78 == 0 and batch_idx!=0:
           writer.add_scalar('train_accuracy', correct/total, epoch)
           writer.add_scalar('train_jaccard', mean(jac_scoreList), epoch)
           print('Train Epoch: {} Loss: {} Acc: {}'.format(epoch, loss.item(), acc))
   test(model, device, test_loader, epoch)
