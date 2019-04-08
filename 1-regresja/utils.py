import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from torch.autograd import Variable


def seed_everything(seed=42):
    '''
    Randomness matters and influences training results. We fix randomness to
    make our code reproductible and be able to compare result between
    different runs.
    '''
    np.random.seed(seed) # cpu vars
    torch.manual_seed(seed) # cpu  vars
    random.seed(seed) # Python


def load_data():
    '''
    Loads boston housting dataset from sklearn.
    Converts to pytorch tensors.
    Wraps with Dataset object.

    :return: (train dataset, test dataset)
    '''
    x_data, y_data = load_boston(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)

    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()

    train_set = torch.utils.data.TensorDataset(x_train, y_train)
    test_set = torch.utils.data.TensorDataset(x_test, y_test)

    return train_set, test_set


def show_visualization(model):
    boston = load_boston()

    features = torch.from_numpy(np.array(boston.data)).float()  # convert the numpy array into torch tensor
    features = Variable(features)                       # create a torch variable and transfer it into GPU

    output = model(features)

    output = output.data.cpu().numpy()
    labels = np.array(boston.target)

    fig, ax = plt.subplots()
    ax.scatter(labels, output)
    ax.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'k--', lw=3)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()
