# -*- coding: utf-8 -*-
import torch
import sys
sys.path.append('../')
import models
from models import Variable
import matplotlib as mpl
from sklearn.utils import shuffle
import ds_readers as ds
import numpy as np

use_cuda = torch.cuda.is_available()

if use_cuda:
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.ioff()  # http://matplotlib.org/faq/usage_faq.html (interactive mode)
else:
    import matplotlib.pyplot as plt

x_dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
y_dtype = torch.cuda.LongTensor if use_cuda else torch.LongTensor

n_epochs = 3


def train_cnn(net, X_train, Y_train, X_test, Y_test):
    if (use_cuda):
        net.cuda()

    print("use_cuda:",use_cuda)
    print("epochs:",n_epochs)
    """ 
    Define a criterion for the neural network - 
    use mean squared error.
    """
    criterion = torch.nn.CrossEntropyLoss()

    """
    Train the network using Adam optimizer. 
    Train the network and test it for each line in the dataset 
    """
    optimizer = torch.optim.Adam(net.parameters(), lr=6e-5)

    n_train = len(X_train)
    X_train = torch.from_numpy(X_train).type(x_dtype)
    Y_train = torch.from_numpy(Y_train).type(y_dtype)
    X_train = X_train.view(n_train, 1, 1, 431, 575)

    n_test = len(X_test)
    X_test = torch.from_numpy(X_test).type(x_dtype)
    Y_test = torch.from_numpy(Y_test).type(y_dtype)
    X_test = X_test.view(n_test, 1, 1, 431, 575)

    log = open("cnn_loss.txt", "w+")

    train_loss_curve = []
    test_loss_curve = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss_mean = []
        #Train the model
        idx = shuffle(range(n_train))
        print("cnn train size:",n_train)
        for i in idx :
            data_point = i

            x_var = Variable(X_train[data_point])
            y_var = Variable(Y_train[data_point])

            optimizer.zero_grad()
            y_pred = net(x_var)

            y_var.unsqueeze_(dim=0)
            train_loss = criterion(y_pred, y_var)
            train_loss.backward()
            optimizer.step()
            epoch_loss_mean.append(train_loss)

        #compute train loss
        train_total_loss = sum(epoch_loss_mean) / len(epoch_loss_mean)
        train_loss_curve.append(train_total_loss.item())
        print("Epoch: {0}, Train Loss: {1}, ".format(epoch, train_total_loss.item()))
        log.write("Epoch: {0}, Train Loss: {1}, \n".format(epoch, train_total_loss.item()))

        #compute test loss
        model.eval()
        epoch_loss_mean = []
        for data_point in range(n_test):
            x_var = Variable(X_test[data_point])
            y_var = Variable(Y_test[data_point])
            y_var.unsqueeze_(dim=0)
            y_pred = model(x_var)
            test_loss = criterion(y_pred, y_var)
            epoch_loss_mean.append(test_loss)
        test_total_loss = sum(epoch_loss_mean) / len(epoch_loss_mean)
        test_loss_curve.append(test_total_loss.item())
        log.write("Epoch: {0}, Test Loss: {1},\n ".format(epoch, test_total_loss.item()))
        print("Epoch: {0}, Test Loss: {1}, ".format(epoch, test_total_loss.item()))

    log.close()

    return train_loss_curve, test_loss_curve

model = models.cnnAlexNet()

"""
# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor])

"""

#fixation_df = ds.get_fixationMap_df()
print("prepare training and test sets")
x_train, x_test, y_train, y_test = ds.get_train_test_dataset()

print("start train")
train_loss_curve, test_loss_curve = train_cnn(model, x_train, y_train, x_test, y_test)

torch.save(model.state_dict(), models.checkpoint_cnn_path())

plt.title('Loss function value for test and train after each epoch')
plt.xlabel('epoch')
plt.ylabel('loss')

epochs = list(range(1, n_epochs + 1))
plt.plot(epochs, train_loss_curve, 'b', label='Q1 Train Data')
plt.plot(epochs, test_loss_curve, 'r', label='Q1 Test Data')

plt.legend(loc='best')
plt.savefig('cnn_train_test_plot.png')

# decide number of epochs
# add batch run - use small batch size to overcome the memory issue
# check val test when ..