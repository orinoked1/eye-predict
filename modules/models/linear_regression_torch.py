import numpy as np
import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
import random


"""
# create dummy data for training
#x_values = [i for i in range(10)]
llist = []
for j in range(10):
    listvals = []
    #x=(j)*100
    #xx = (j+1)*100
    for i in range(100):
        x = random.random()
        listvals.append((x, x+1))
    llist.append(listvals)
x_values = llist
x_train = np.array(x_values, dtype=np.float32)
# Normelize
x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
a=x_train[0]
b=x_train[5]
#x_train = x_train.reshape(1,1)

y_values = [i for i in range(10)]
y_train = np.array(y_values, dtype=np.float32)
# Normelize
y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min())
y_train = y_train.reshape(-1, 1)
"""


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out

class linearRegressionModel:
    def __init__(self, seed, run_name, stim_size, scanpath_lan, stimType, bin_count):
        # fix random seed for reproducibility
        #self.seed = seed
        #numpy.random.seed(self.seed)
        self.model = None
        self.history = None
        #self.scanpath_lan = scanpath_lan
        #self.run_name = run_name
        self.batch_size = 128
        self.num_epochs = 100
        #self.stimSize = stim_size
        self.num_class = 1
        self.LR = 0.001
        self.optimizer = None
        self.loss_function = None
        #self.stimType = stimType
        self.input_type = 'scanpath'
        self.datapath = "/export/home/DATA/schonberglab/pycharm_eyePredict/etp_data/processed/"
        self.corr = None
        #self.bin_count = bin_count
        self.train_loss_curve = []
        self.predicted = None

    def define_model(self, trainX):
        inputDim = trainX.shape[1]*2  # takes variable 'x'
        outputDim = 1  # takes variable 'y'
        learningRate = 0.001
        epochs = 100

        self.model = linearRegression(inputDim, outputDim)
        ##### For GPU #######
        if torch.cuda.is_available():
            print("use GPU")
            self.model.cuda()

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learningRate)

        return

    def train_model(self, x_train, y_train, valX, valY):
        a=x_train[0]
        b=x_train[7]

        for epoch in range(self.num_epochs):
            # Converting inputs and labels to Variable
            if torch.cuda.is_available():
                dtype = torch.cuda.FloatTensor
                inputs = Variable(torch.Tensor(x_train).type(dtype)) #torch.from_numpy(x_train).cuda()
                labels = Variable(torch.Tensor(y_train).type(dtype)) #torch.from_numpy(y_train).cuda.FloatTensor)
            else:
                dtype = torch.FloatTensor
                inputs = Variable(torch.Tensor(x_train).type(dtype)) #torch.from_numpy(x_train))
                labels = Variable(torch.Tensor(y_train).type(dtype)) #torch.from_numpy(y_train))

            # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
            self.optimizer.zero_grad()

            # get output from the model, given the inputs
            outputs = self.model(inputs)

            # get loss for the predicted output
            loss = self.criterion(outputs, labels)
            print(loss)
            # get gradients w.r.t to parameters
            loss.backward()

            # update parameters
            self.optimizer.step()

            print('epoch {}, loss {}'.format(epoch, loss.item()))
            # data for convergence plot
            self.train_loss_curve.append(loss.item())

    def test_model(self, x_train, testY):

        with torch.no_grad(): # we don't need gradients in the testing phase
            if torch.cuda.is_available():
                dtype = torch.cuda.FloatTensor
                self.predicted = self.model(torch.Tensor(x_train).type(dtype)).cpu().data.numpy()
            else:
                dtype = torch.FloatTensor
                self.predicted = self.model(torch.Tensor(x_train).type(dtype)).data.numpy()
            print(self.predicted)

        return

    def metrices(self, y_train):

        epochs = list(range(1, self.num_epochs + 1))

        fig = plt.figure(1)
        plt.clf()
        plt.plot(y_train, 'go', label='True data', alpha=0.5)
        plt.plot(self.predicted, '--', label='Predictions', alpha=0.5)
        plt.legend(loc='best')
        fig.savefig("/export/home/DATA/schonberglab/pycharm_eyePredict/etp_data/processed/figs/torch/linear_regression/" + "fig.pdf", bbox_inches='tight')

        fig = plt.figure(2)
        plt.title('Loss function value for train after each epoch')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(epochs, self.train_loss_curve,'b', label='Train Data Loss')
        plt.legend(loc='best')
        plt.show
        plt.savefig("/export/home/DATA/schonberglab/pycharm_eyePredict/etp_data/processed/figs/torch/linear_regression/" + 'Loss_function.pdf', bbox_inches='tight')


#linearReg = linearRegressionModel()#seed, run_name, stim_size, scanpath_lan, stimType, bin_count)
# Build train and evaluate model
#linearReg.define_model()
#linearReg.train_model()#trainX, trainY, valX, valY)
#linearReg.test_model()#valX, valY)
#linearReg.metrices()