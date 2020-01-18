import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from PIL import Image
import torch 
import torch.nn as nn 
import torchvision 
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import glob
import os
from model import Classification
from data import CatDogDataset
from torchvision import transforms


image_size = (100, 100)
image_row_size = image_size[0] * image_size[1]
image_size = (100, 100)
image_row_size = image_size[0] * image_size[1]


mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
transform = transforms.Compose([
                                transforms.Resize(image_size), 
                                # transforms.Grayscale(),
                                transforms.ToTensor(), 
                                transforms.Normalize(mean, std)])


path = '/home/aims/Documents/Pytorch/pytorch_exercise/data'

net =  Classification()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


train_data = CatDogDataset(path+"/"+'train', transform=transform)
test_data = CatDogDataset(path+"/"+'val',transform=transform)

trainloader = torch.utils.data.DataLoader(test_data, batch_size=64,
                                          shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64,
                                         shuffle=True, num_workers=4)
#Training

for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        if i % 100 == 0:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss/100))
            running_loss = 0.0
#Testing

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader :
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
