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

class cnnSimple(torch.nn.Module):
    def __init__(self):
        # Our input shape is (1, 431, 575)
        super(cnnSimple, self).__init__()

        # Input channels = 1, output channels = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=11, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Input channels = 16, output channels = 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 4*4*16 input features, 16 output features (see sizing flow below)
        self.fc1 = nn.Linear(24 * 33 * 32, 16)

        # dropout of 50% before the last layer
        self.dropout = nn.Dropout(0.5)

        # 16 input features, 11 output features for our 10 defined classes
        self.fc2 = nn.Linear(16, 11)

    def forward(self, x):
        # Computes the activation of the first convolution
        # Size changes from (1, 431, 575) to (130, 141, 16)
        x = F.relu(self.conv1(x))

        # Size changes from (130, 141, 16) to (65, 70, 16)
        x = self.pool1(x)

        # Computes the activation of the first convolution
        # Size changes from (65, 70, 16) to (62, 67, 32)
        x = F.relu(self.conv2(x))

        # Size changes from (62, 67, 32) to (31, 33, 32)
        x = self.pool2(x)

        # Reshape data to input to the input layer of the neural net
        # Size changes from (31, 33, 32) to (1, 25344)
        # Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 24 * 33 * 32)

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


def checkpoint_cnnSimple_path():
    model_path = "../cnnSimple-net-model.checkpoint"
    if (use_cuda):
        model_path = "../cnnSimple-net-model-cuda.cheakpoint"

    return model_path