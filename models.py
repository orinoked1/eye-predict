# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

use_cuda = torch.cuda.is_available()

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if use_cuda:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)


class cnnAlexNet(torch.nn.Module):
    '''
    code from torchvision/models/alexnet.py
    结构参考 <https://arxiv.org/abs/1404.5997>
    '''

    def __init__(self, num_classes=11):
        super(cnnAlexNet, self).__init__()

        self.model_name = 'cnnalexnet'

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(192, 64, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(64, 16, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 12 * 17, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 12 * 17)
        x = self.classifier(x)
        return x

class cnn(torch.nn.Module):
    def __init__(self):
        # Our batch shape for input x is (1, 431, 575)
        super(cnn, self).__init__()

        # Input channels = 1, output channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        # 4*4*16 input features, 16 output features (see sizing flow below)
        self.fc1 = nn.Linear(4 * 4 * 16, 16)

        # 16 input features, 2 output features for our 2 defined classes
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        # Computes the activation of the first convolution
        # Size changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv1(x))

        # Size changes from (18, 32, 32) to (18, 16, 16)
        x = self.pool(x)

        # Reshape data to input to the input layer of the neural net
        # Size changes from (16, 16, 16) to (1, 4608)
        # Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 4 * 4 * 16)

        # Computes the activation of the first fully connected layer
        # Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))

        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 64) to (1, 10)
        x = self.fc2(x)
        return(x)

class FaceSimpleDetect12net(torch.nn.Module):
    def __init__(self):
        # Our batch shape for input x is (3, 12, 12)
        super(FaceSimpleDetect12net, self).__init__()

        # Input channels = 3, output channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        #repleace nn.Linear(4 * 4 * 16, 16) with conv layer
        #we keep the same input channels to match the FC output and change the kernal size to 4 with stribe
        #to make sure we use all the paramters.
        self.conv2 = nn.Conv2d(16, 16, kernel_size=4, stride=1)

        # 16 input features, 2 output features for our 2 defined classes
        self.conv3 = nn.Conv2d(16, 2, kernel_size=1, stride=1)

        #perform soft max to normalize the results
        self.softmax = nn.Softmax()

    def forward(self, x):
        # Computes the activation of the first convolution
        # Size changes from (3, x, x) to (18, x, x)
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        # Computes the activation of the next convolution
        # Size changes from (16 x, x) to (16, x/4, x/4)
        #x = F.relu(self.conv2(x))
        x = self.conv2(x)

        # Computes the activation of the last convolution
        # Size changes from (16 x', x') to (2, x', x')
        #x = F.relu(self.conv3(x))
        x = self.conv3(x)

        x = self.softmax(x)
        return x

class FaceDetect24net(torch.nn.Module):
    def __init__(self):
        # Our batch shape for input x is (3, 24, 24)
        super(FaceDetect24net, self).__init__()

        # Input channels = 3, output channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        # 8 * 8 * 64 input features, 128 output features (see sizing flow below)
        self.fc1 = nn.Linear(9 * 9 * 64, 128)

        # 16 input features, 2 output features for our 2 defined classes
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # Computes the activation of the first convolution
        # Size changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv1(x))

        # Size changes from (18, 32, 32) to (18, 16, 16)
        x = self.pool(x)

        # Reshape data to input to the input layer of the neural net
        # Size changes from (16, 16, 16) to (1, 4608)
        # Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 9 * 9 * 64)

        # Computes the activation of the first fully connected layer
        # Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))

        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 64) to (1, 10)
        x = self.fc2(x)
        return(x)


def checkpoint_cnn_path():
    model_path = "../cnn-model.checkpoint"
    if (use_cuda):
        model_path = "../cnn-model-cuda.cheakpoint"

    return model_path


def checkpoint_24net_path():
    model_path = "../24-net-model.checkpoint"
    if (use_cuda):
        model_path = "../24-net-model-cuda.cheakpoint"

    return model_path