import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import torch
from torch import nn
from torch import optim
import torch.autograd as autograd


use_cuda = torch.cuda.is_available()
print("Log... running on GPU?", use_cuda)

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if use_cuda:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)



