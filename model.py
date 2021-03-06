import torch

import torch.nn as nn
import torch.nn.functional as F


class Classification(nn.Module):
    def __init__(self):
        super(Classification,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3)

       # self.conv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3)
        self.fc1 = nn.Linear(in_features =129600,out_features = 32)
        self.fc2 = nn.Linear(in_features = 32,out_features = 2)


    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.relu(self.conv2(x))
        x =F.relu(self.conv3(x))

        # print(self.num_flat_features(x))
        x = x.view(-1,self.num_flat_features(x))

        x = F.relu(self.fc1(x))

        #x = self.fc2(x)
        x = F.softmax(self.fc2(x), dim=1)

        return x

    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        
        return num_features
