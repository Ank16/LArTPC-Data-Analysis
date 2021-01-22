#Importing the modules we need
import torch
import spconv
import torch.utils.data as data_utils
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import argparse
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#Setting log file directory
writer = SummaryWriter('logs/fit/')

#Function we use to split 5 particle data into EM shower vs non-EM shower/Track particles
def particleToEMTrack(inputArray):
   for i in range(inputArray.shape[0]):
       v=inputArray[i]
       if v==0 or v==1:
           inputArray[i] = 0
       else:
           inputArray[i] = 1
  
   return inputArray

#Load the data into array variables
X = np.load('X64.npy')
X = X/225.0
X = np.rollaxis(X, 3, 1)
y = np.load('train_y.npy')
y = np.argmax(y, axis=1)
y = particleToEMTrack(y)
print(X.shape)
print(y.shape)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#Convert data into tenors (this is done because PyTorch reads data through tensors)
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)
X_val = torch.from_numpy(X_val)
y_val = torch.from_numpy(y_val)
X_train = X_train.float()
X_val = X_val.float()

#Make datasets for training and testing/validation out of the data
train_data = data_utils.TensorDataset(X_train, y_train)
val_data = data_utils.TensorDataset(X_val, y_val)

#Class that defines the model
class Net(nn.Module):
   def __init__(self):
       super(Net, self).__init__()
       self.net = spconv.SparseSequential(
           nn.BatchNorm1d(1),
           spconv.SubMConv2d(1, 64, 3, 1),
           nn.BatchNorm1d(64),
           nn.ReLU(),
           spconv.SubMConv2d(64, 64, 3, 1),
           nn.BatchNorm1d(64),
           nn.ReLU(),
           spconv.SparseMaxPool2d(2, 2),
           nn.Dropout2d(0.25),
          
           spconv.SubMConv2d(64, 128, 3, 1),
           nn.BatchNorm1d(128),
           nn.ReLU(),
           spconv.SubMConv2d(128, 128, 3, 1),
           nn.BatchNorm1d(128),
           nn.ReLU(),
           spconv.SparseMaxPool2d(2, 2),
           nn.Dropout2d(0.25),
          
           spconv.SubMConv2d(128, 256, 3, 1),
           nn.BatchNorm1d(256),
           nn.ReLU(),
           spconv.SubMConv2d(256, 256, 3, 1),
           nn.BatchNorm1d(256),
           nn.ReLU(),
           spconv.SparseMaxPool2d(2, 2),
           nn.Dropout2d(0.25),
          
           spconv.ToDense(),
       )
       self.fc1 = nn.Linear(16384, 128)
       self.fc2 = nn.Linear(128, 256)
       self.fc3 = nn.Linear(256, 256)
       self.fc4 = nn.Linear(256, 2)
       self.dropout1 = nn.Dropout2d(0.25)
       self.dropout2 = nn.Dropout2d(0.5)
 
 
   def forward(self, x: torch.Tensor):
       x_sp = spconv.SparseConvTensor.from_dense(x.reshape(-1, 64, 64, 1))
       x = self.net(x_sp)
       x = torch.flatten(x, 1)
       x = self.dropout1(x)
       x = self.fc1(x)
       x = F.relu(x)
       x = self.fc2(x)
       x = self.dropout2(x)
       x = self.fc3(x)
       x = self.fc4(x)
       output = F.log_softmax(x, dim=1)
       return output

#Custom training loop function
def train(args, model, device, train_loader, optimizer, epoch):
   model.train()
   global correct
   correct = 0
   for batch_idx, (data, target) in enumerate(train_loader):
       data, target = data.to(device), target.to(device)
       optimizer.zero_grad()
       output = model(data)
       pred = output.argmax(dim=1, keepdim=True)
       correct += pred.eq(target.view_as(pred)).sum().item()
       loss = F.cross_entropy(output, target.long())
       pred = output.argmax(dim=1, keepdim=True)
       f1 = f1_score(target.cpu(), pred.cpu(), average='weighted')
       loss.backward()
       optimizer.step()
       if batch_idx % 312 == 0 and batch_idx!=0:
           writer.add_scalar('train_f1', f1, epoch)
           writer.add_scalar('train_accuracy', correct / len(train_loader.dataset), epoch)
           print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
               epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item()))

#Custom testing/validation function
def test(model, device, test_loader, epoch):
   model.eval()
   global test_loss
   test_loss = 0
   global correct2
   correct2 = 0
   with torch.no_grad():
       for data, target in test_loader:
           data, target = data.to(device), target.to(device)
           global output
           output = model(data)
           test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
           pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
           global report
           report = classification_report(target.cpu(), pred.cpu())
           correct2 += pred.eq(target.view_as(pred)).sum().item()
           f1 = f1_score(target.cpu(), pred.cpu(), average='weighted')
 
   test_loss /= len(test_loader.dataset)
  
   writer.add_scalar('val_f1', f1, epoch)
   writer.add_scalar('val_accuracy', correct2 / len(test_loader.dataset), epoch)
 
   print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
       test_loss, correct2, len(test_loader.dataset),
       100. * correct2 / len(test_loader.dataset)))

#Main function which trains and tests the model and puts logs into log files
def main():
   # Training settings
   parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
   parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                       help='input batch size for training (default: 64)')
   parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                       help='input batch size for testing (default: 1000)')
   parser.add_argument('--epochs', type=int, default=50, metavar='N',
                       help='number of epochs to train (default: 14)')
   parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                       help='learning rate (default: 1.0)')
  
   args = parser.parse_args(args=[])
   use_cuda = not args.no_cuda and torch.cuda.is_available()
 
   torch.manual_seed(args.seed)
 
   device = torch.device("cuda" if use_cuda else "cpu")
 
   kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
   train_loader = torch.utils.data.DataLoader(train_data,
       batch_size=args.batch_size, shuffle=True, **kwargs)
   test_loader = torch.utils.data.DataLoader(val_data,
       batch_size=args.test_batch_size, shuffle=False, **kwargs)
  
   global model
   model = Net().to(device)
   optimizer = optim.Adam(model.parameters(), lr=args.lr)
 
   scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
   for epoch in range(1, args.epochs + 1):
       train(args, model, device, train_loader, optimizer, epoch)
       test(model, device, test_loader, epoch)
       scheduler.step(test_loss)
 
   if args.save_model:
       torch.save(model.state_dict(), "mnist_cnn.pt")

#Train the model by running main function
main()
