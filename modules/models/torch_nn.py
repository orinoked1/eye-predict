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

class TorchNn:
    def __init__(self, seed, train, val, run_name, stim_size):
        # fix random seed for reproducibility
        np.random.seed(seed)
        self.data = dataset

        input_dim = X[1].size
        hidden_dim = 100
        output_dim = 30

        n = X.shape[0]

        # Allocating 80% of the data for train and 20% to test
        train_ratio = 0.8
        train_size = round(train_ratio * n)
        test_size = n - train_size
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        epochs = list(range(1, max_epochs + 1))

        # tansform to tansors
        X_train, X_test = torch.Tensor(X_train).type(dtype), torch.Tensor(X_test).type(dtype)
        y_train, y_test = torch.Tensor(y_train).type(dtype), torch.Tensor(y_test).type(dtype)

        model = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                              nn.ReLU(),
                              nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                          nn.ReLU(),
                          nn.Linear(hidden_dim, output_dim))
        self.history = None
        self.run_name = run_name
        self.seed = seed
        self.batch_size = 32
        self.num_epochs = 100
        self.stimSize = stim_size
        self.num_class = 1



    def load(self):
        df = self.data

        X = X.astype(np.float32)

        y = df[df.columns[:-1]].values
        # scale target coordinates to [-1, 1]
        y = (y - 48) / 48
        X, y = shuffle(X, y)
        y = y.astype(np.float32)

        return X, y


    def train(self, n, model, optimizer, criterion, epoch):
        X, y = self.load()
        model = model.train()
        # Shuffle the data between epochs.
        idx = shuffle(range(n))
        lossMean = []
        for i in idx:
            x_var = Variable(x[i])
            y_var = Variable(y[i])

            optimizer.zero_grad()
            y_pred = model(x_var)
            loss = criterion(y_pred, y_var)
            loss.backward()
            optimizer.step()
            lossMean.append(loss)
        return (sum(lossMean) / len(lossMean))


    def test(x, y, n, model, epoch):
        model = model.eval()
        criterion = nn.MSELoss()
        lossMean = []
        with torch.no_grad():
            for i in range(n):
                x_var = Variable(x[i])
                y_var = Variable(y[i])

                y_pred = model(x_var)
                loss = criterion(y_pred, y_var)
                lossMean.append(loss)
        return (sum(lossMean) / len(lossMean))


    def train_test_curve(model, log, epochs, X_train, y_train, train_size,
                         X_test, y_test, test_size):
        if (use_cuda):
            model.cuda()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())

        train_loss_curve = []
        test_loss_curve = []

        for epoch in epochs:
            train_loss = train(X_train, y_train, train_size, model, optimizer, criterion, epoch)
            train_loss_curve.append(train_loss.item())
            log.write("Epoch: {0}, Train Loss: {1}, ".format(epoch, train_loss.item()))
            print("Epoch: {0}, Train Loss: {1}, ".format(epoch, train_loss.item()))

            test_loss = test(X_test, y_test, test_size, model, epoch)
            test_loss_curve.append(test_loss.item())
            log.write("Epoch: {0}, Test Loss: {1}, ".format(epoch, test_loss.item()))
            print("Epoch: {0}, Test Loss: {1}, ".format(epoch, test_loss.item()))

        # loss function value (train and test) at the final epoch.
        log.write("Final Epoch: {0}, Final Train Loss: {1}, ".format(max_epochs, train_loss.item()))
        print("Final Epoch: {0}, Final Train Loss: {1}, ".format(max_epochs, train_loss.item()))
        log.write("Final Epoch: {0}, Final Test Loss: {1}, ".format(max_epochs, test_loss.item()))
        print("Final Epoch: {0}, Final Test Loss: {1}, ".format(max_epochs, test_loss.item()))

        return train_loss_curve, test_loss_curve


    class Reshape(nn.Module):
        def __init__(self, *args):
            super(Reshape, self).__init__()
            self.shape = args

        def forward(self, x):
            return x.view(self.shape)


    input_dim = X[1].size
    hidden_dim = 100
    output_dim = 30

    n = X.shape[0]

    # Allocating 80% of the data for train and 20% to test
    train_ratio = 0.8
    train_size = round(train_ratio * n)
    test_size = n - train_size
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    epochs = list(range(1, max_epochs + 1))

    # tansform to tansors
    X_train, X_test = torch.Tensor(X_train).type(dtype), torch.Tensor(X_test).type(dtype)
    y_train, y_test = torch.Tensor(y_train).type(dtype), torch.Tensor(y_test).type(dtype)

    model = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                          nn.ReLU(),
                          nn.Linear(hidden_dim, output_dim))

    train_loss_curve, test_loss_curve = train_test_curve(model, log, epochs,
                                                         X_train, y_train,
                                                         train_size, X_test,
                                                         y_test, test_size)

    # Plot the cost function value for test and train after each epoch.


    plt.title('Loss function value for test and train after each epoch')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xlim(2, max_epochs)
    plt.ylim(0, train_loss_curve[1])

    # fig, ax = plt.subplots()
    plt.plot(epochs, train_loss_curve, 'b', label='Q3.2 Train Data')
    plt.plot(epochs, test_loss_curve, 'r', label='Q3.2 Test Data')
    # start, end = ax.get_ylim()
    # ax.yaxis.set_ticks(np.arange(start, end, 0.712123))
    # ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.003f'))


    plt.legend(loc='best')
    # plt.show
    plt.savefig('Q3_2_plot.png')




