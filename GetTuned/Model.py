"""
    @file:              Model.py
    @Author:            Nicolas Raymond
                        Alexandre Ayotte
    @Creation Date:     30/09/2019
    @Last modification: 25/11/2019

    @Reference: 1)  K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.

    @Description:       This program generates models of different types to use for our classification problems.
"""

from .DataManager import create_dataloader, validation_split, dataset_to_loader
from .Module import Swish, Mish, ResModuleV1, ResModuleV2
import numpy as np
import sklearn.svm as svm
import sklearn.neural_network as nn
import matplotlib.pyplot as plt
import torch
import time
from sklearn.model_selection import train_test_split
from enum import Enum, unique
from tqdm import tqdm

@unique
class HPtype(Enum):

    """
    Class containing possible types of hyper-parameters
    """

    real = 1
    integer = 2
    categorical = 3


class Hyperparameter:

    def __init__(self, name, type, value=None):

        """
        Class that defines an hyper-parameter

        :param name: Name of the hyper-parameter
        :param type: One type out of HPtype (real,.integer, categorical)
        :param value: List with the value of the hyper-parameter
        """

        self.name = name
        self.type = type
        self.value = value


class Model:

    def __init__(self, HP_Dict):

        """
        Class that generates a model to solve a classification problem

        :param HP_Dict: Dictionary containing all hyper-parameters of the model
        """

        self.HP_space = HP_Dict

    def set_hyperparameters(self, hyperparams):

        """
        Change hyper-parameters of our model

        Note that it will be override by the children's classes

        :param hyperparams: Dictionary of hyper-parameters to change
        """

        raise NotImplementedError

    def fit(self, X_train=None, t_train=None, dtset=None):

        """
        Train our model

        Note that it will be override by the children's classes

        :param X_train: NxD numpy array of observations {N : nb of obs, D : nb of dimensions}
        :param t_train: Nx1 numpy array of classes associated with each observation
        :param dtset: A torch dataset which contain our train data points and labels
        """

        raise NotImplementedError

    def set_max_epoch(self, max_epoch):

        """

        Set maximal number of epochs to do in training

        :param max_epoch: maximum number of epochs to do during training
        """
        raise NotImplementedError

    def predict(self, X):

        """
        Predict classes for our observations in the input array X

        Note that it will be override by the children's classes

        :param X: NxD numpy array of observations {N : nb of obs, D : nb of dimensions}
        :return: Nx1 numpy array of classes predicted for each observation
        """

        raise NotImplementedError

    def score(self, X=None, t=None, dtset=None):

        """
        Compute the model accuracy over a given test dataset.

        Note that it will be override by the children's classes

        :param X: NxD numpy array of observations {N : nb of obs, D : nb of dimensions}
        :param t: Nx1 numpy array of classes associated with each observation
        :param dtset: A torch dataset which contain our test data points and labels
        :return: Good classification rate
        """

        raise NotImplementedError

    def cross_validation(self, X_train=None, t_train=None, dtset=None, valid_size=0.2, nb_of_cross_validation=3):

        """
        Compute a cross validation over a given dataset and calculate the average accuracy

        :param X_train: NxD numpy array of observations {N : nb of obs, D : nb of dimensions}
        :param t_train: Nx1 numpy array of classes associated with each observation
        :param dtset: A torch dataset which contain our train data points and labels
        :param valid_size: Proportion of the dataset that will be use as validation data
        :param nb_of_cross_validation:  Number of data splits and validation to execute
        :return: Mean of score (accuracy)
        """

        res = np.array([])

        for i in range(nb_of_cross_validation):

            if not(X_train is None or t_train is None):
                x_train, x_valid, y_train, y_valid = train_test_split(X_train, t_train, test_size=valid_size)
                self.fit(X_train=x_train, t_train=y_train)
                res = np.append(res, self.score(X=x_valid, t=y_valid))

            elif not(dtset is None):
                d_train, d_valid = validation_split(dtset=dtset, valid_size=valid_size)
                self.fit(dtset=d_train)
                res = np.append(res, self.score(dtset=d_valid))

            else:
                raise Exception("Features or labels missing. X is None: {}, t is None: {}, dtset is None: {}".format(
                    X_train is None, t_train is None, dtset is None))

        return np.mean(res)

    def plot_data(self, data, classes):

        """
        Plot data points and spaces of separation done by the model for 2D cases only.

        :param data: Nx2 numpy array of observations {N : nb of obs}
        :param classes: Classes associate with each data point.
        """

        if data.shape[1] != 2:
            raise Exception('Method only available for 2D plotting (two dimensions datasets)')

        else:
            ix = np.arange(data[:, 0].min(), data[:, 0].max(), 0.01)
            iy = np.arange(data[:, 1].min(), data[:, 1].max(), 0.01)
            iX, iY = np.meshgrid(ix, iy)
            x_vis = np.hstack([iX.reshape((-1, 1)), iY.reshape((-1, 1))])
            contour_out = self.predict(x_vis)
            contour_out = contour_out.reshape(iX.shape)

            plt.contourf(iX, iY, contour_out)
            plt.scatter(data[:, 0], data[:, 1], s=105, c=classes, edgecolors='b')
            plt.title('Accuracy on test : {} %'.format(self.score(X=data, t=classes)*100))
            plt.show()


class SVM(Model):

    def __init__(self, C=1.0, kernel="rbf", degree=3, gamma='auto', coef0=0.0, max_iter=1000):

        """
        Class that generates a support vector machine

        Some hyper-parameters are conditional to others!
        Take a look at https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC for
        more information on hyper-parameters

        :param C: Penalty parameter C of the error term
        :param kernel: Kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’ or ‘sigmoid’
        :param degree: Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels
        :param gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’
        :param coef0: Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
        """

        self.model_frame = svm.SVC(C, kernel, degree, gamma, coef0, max_iter=max_iter)

        if kernel == 'rbf':
            super(SVM, self).__init__({'C': Hyperparameter('C', HPtype.real, [C]),
                                       'kernel': Hyperparameter('kernel', HPtype.categorical, [kernel]),
                                       'gamma': Hyperparameter('gamma', HPtype.real, [gamma])})

        elif kernel == 'linear':
            super(SVM, self).__init__({'C': Hyperparameter('C', HPtype.real, [C]),
                                       'kernel': Hyperparameter('kernel', HPtype.categorical, [kernel])})

        elif kernel == 'poly':
            super(SVM, self).__init__({'C': Hyperparameter('C', HPtype.real, [C]),
                                       'kernel': Hyperparameter('kernel', HPtype.categorical, [kernel]),
                                       'degree': Hyperparameter('degree', HPtype.integer, [degree]),
                                       'gamma': Hyperparameter('gamma', HPtype.real, [gamma]),
                                       'coef0': Hyperparameter('coef0', HPtype.real, [coef0])})

        elif kernel == 'sigmoid':
            super(SVM, self).__init__({'C': Hyperparameter('C', HPtype.real, [C]),
                                       'kernel': Hyperparameter('kernel', HPtype.categorical, [kernel]),
                                       'gamma': Hyperparameter('gamma', HPtype.real, [gamma]),
                                       'coef0': Hyperparameter('coef0', HPtype.real, [coef0])})

        else:
            raise Exception('No such kernel ("{}") implemented'.format(kernel))

    def fit(self, X_train=None, t_train=None, dtset=None):

        """
        Train our model

        :param X_train: NxD numpy array of the observations of the training set {N : nb of obs, D : nb of dimensions}
        :param t_train: Nx1 numpy array of classes associated with each observation in the training set
        :param dtset: A torch dataset which contain our train data points and labels
        """

        if X_train is None or t_train is None:
            if dtset is None:
                raise Exception("Features or labels missing. X is None: {}, t is None: {}, dtset is None: {}".format(
                    X_train is None, t_train is None, dtset is None))
            else:
                X_train = dtset.data
                t_train = dtset.targets

        self.model_frame.fit(X_train, t_train)

    def predict(self, X):

        """
        Predict classes for our observations in the input array X

        :param X: NxD numpy array of observations {N : nb of obs, D : nb of dimensions}
        :return: Nx1 numpy array of classes predicted for each observation
        """

        return self.model_frame.predict(X)

    def score(self, X=None, t=None, dtset=None):

        """
        Compute the model accuracy over a given test dataset.

        :param X: NxD numpy array of observations {N : nb of obs, D : nb of dimensions}
        :param t: Nx1 numpy array of classes associated with each observation
        :param dtset: A torch dataset which contain our test data points and labels
        :return: Good classification rate
        """

        if X is None or t is None:
            if dtset is None:
                raise Exception("Features or labels missing. X is None: {}, t is None: {}, dtset is None: {}".format(
                    X is None, t is None, dtset is None))
            else:
                X = dtset.data
                t = dtset.targets

        predictions = self.predict(X)

        diff = t - predictions

        return ((diff == 0).sum()) / len(diff)  # (Nb of good predictions / nb of predictions)

    def set_hyperparameters(self, hyperparams):

        """
        Change hyper-parameters of our model

        :param hyperparams: Dictionary of hyper-parameters to change
        """

        self.model_frame.set_params(**hyperparams)

    def set_max_epoch(self, max_epoch):

        """
        Set parameter max_iter in our model

        :param max_epoch: maximum number of epochs to do during training
        """

        self.model_frame.set_params(**{'max_iter': max_epoch})


class MLP(Model):

    def __init__(self, hidden_layers_number=1, layers_size=100, activation='relu', solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200,
                 momentum=0.9, beta_1=0.9, beta_2=0.999):

        """
        Class that generates a Multi-layer Perceptron classifier.

        This model optimizes the log-loss function using LBFGS or stochastic gradient descent.

        Some hyper-parameters are conditional to others!
        For more information take a look at :
        https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier

        :param hidden_layers_number: Number of hidden layers in the network
        :param layers_size: Number of neurons per layer (equal for every layer)
        :param activation: Activation function for the hidden layer {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
        :param solver: The solver for weight optimization {‘lbfgs’, ‘sgd’, ‘adam’}
        :param alpha: L2 penalty (regularization term) parameter
        :param batch_size: Size of minibatches for stochastic optimizers.
        :param learning_rate: Learning rate schedule for weight updates {‘constant’, ‘invscaling’, ‘adaptive’}
        :param learning_rate_init: The initial learning rate used
        :param power_t: The exponent for inverse scaling learning rate.
        :param max_iter: Maximum number of iterations
        :param momentum: Momentum for gradient descent update
        :param beta_1: Exponential decay rate for estimates of first moment vector in adam, should be in [0, 1)
        :param beta_2: Exponential decay rate for estimates of second moment vector in adam, should be in [0, 1)
        """

        hidden_layer_sizes = tuple([layers_size]*hidden_layers_number)

        self.model_frame = nn.MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver,
                                            alpha=alpha, batch_size=batch_size, learning_rate=learning_rate,
                                            learning_rate_init=learning_rate_init, power_t=power_t, max_iter=max_iter,
                                            momentum=momentum, beta_1=beta_1, beta_2=beta_2)

        if solver == 'adam':
            super(MLP, self).__init__(
                {'hidden_layers_number': Hyperparameter('hidden_layers_number', HPtype.integer, [hidden_layers_number]),
                 'layers_size': Hyperparameter('layers_size', HPtype.integer, [layers_size]),
                 'activation': Hyperparameter('activation', HPtype.categorical, [activation]),
                 'solver': Hyperparameter('solver', HPtype.categorical, [solver]),
                 'alpha': Hyperparameter('alpha', HPtype.real, [alpha]),
                 'batch_size': Hyperparameter('batch_size', HPtype.integer, [batch_size]),
                 'learning_rate_init': Hyperparameter('learning_rate_init', HPtype.real, [learning_rate_init]),
                 'beta_1': Hyperparameter('beta_1', HPtype.real, [beta_1]),
                 'beta_2': Hyperparameter('beta_2', HPtype.real, [beta_2])})

        elif solver == 'sgd':
            super(MLP, self).__init__(
                {'hidden_layers_number': Hyperparameter('hidden_layers_number', HPtype.integer, [hidden_layers_number]),
                 'layers_size': Hyperparameter('layers_size', HPtype.integer, [layers_size]),
                 'activation': Hyperparameter('activation', HPtype.categorical, [activation]),
                 'solver': Hyperparameter('solver', HPtype.categorical, [solver]),
                 'alpha': Hyperparameter('alpha', HPtype.real, [alpha]),
                 'batch_size': Hyperparameter('batch_size', HPtype.integer, [batch_size]),
                 'learning_rate': Hyperparameter('learning_rate', HPtype.categorical, [learning_rate]),
                 'learning_rate_init': Hyperparameter('learning_rate_init', HPtype.real, [learning_rate_init]),
                 'power_t': Hyperparameter('power_t', HPtype.real, [power_t]),
                 'momentum': Hyperparameter('momentum', HPtype.real, [momentum])})

        elif solver == 'lbfgs':
            super(MLP, self).__init__(
                {'hidden_layers_number': Hyperparameter('hidden_layers_number', HPtype.integer, [hidden_layers_number]),
                 'layers_size': Hyperparameter('layers_size', HPtype.integer, [layers_size]),
                 'activation': Hyperparameter('activation', HPtype.categorical, [activation]),
                 'solver': Hyperparameter('solver', HPtype.categorical, [solver]),
                 'alpha': Hyperparameter('alpha', HPtype.real, [alpha])})

    def fit(self, X_train=None, t_train=None, dtset=None):

        """
        Train our model

        :param X_train: NxD numpy array of the observations of the training set
        :param t_train: Nx1 numpy array classes associated with each observations in the training set
        :param dtset: A torch dataset which contain our train data points and labels
        """

        if X_train is None or t_train is None:
            if dtset is None:
                raise Exception("Features or labels missing. X is None: {}, t is None: {}, dtset is None: {}".format(
                    X_train is None, t_train is None, dtset is None))
            else:
                X_train = dtset.data
                t_train = dtset.targets

        self.model_frame.fit(X_train, t_train)

    def predict(self, X):

        """
        Predict classes for our observations in the input array X

        :param X: NxD numpy array of observations {N : nb of obs, D : nb of dimensions}
        :return: Nx1 numpy array classes predicted for each observation
        """

        return self.model_frame.predict(X)

    def score(self, X=None, t=None, dtset=None):

        """
        Compute the model accuracy over a given test dataset.

        :param X: NxD numpy array of observations {N : nb of obs, D : nb of dimensions}
        :param t: Nx1 numpy array of classes associated with each observation
        :param dtset: A torch dataset which contain our test data points and labels
        :return: Good classification rate
        """

        if X is None or t is None:
            if dtset is None:
                raise Exception("Features or labels missing. X is None: {}, t is None: {}, dtset is None: {}".format(
                    X is None, t is None, dtset is None))
            else:
                X = dtset.data
                t = dtset.targets

        predictions = self.predict(X)

        diff = t - predictions

        return ((diff == 0).sum()) / len(diff)  # (Nb of good predictions / nb of predictions)

    def set_hyperparameters(self, hyperparams):

        """
        Change hyper-parameters of our model

        :param hyperparams: Dictionary of hyper-parameters to change
        """

        # We do a copy of the dict to avoid changing the original dict
        hps = hyperparams.copy()

        # We compute the hidden layer sizes parameter to fit with sklearn MLP while removing hln and ls
        # hyper-parameters from the dictionary
        hidden_layers_number = hps.pop('hidden_layers_number', self.HP_space['hidden_layers_number'].value[0])
        layers_size = hps.pop('layers_size', self.HP_space['layers_size'].value[0])
        hidden_layer_sizes = tuple([layers_size]*hidden_layers_number)

        # We add hidden_layer_sizes to the dict
        hps['hidden_layer_sizes'] = hidden_layer_sizes

        self.model_frame.set_params(**hps)

    def set_max_epoch(self, max_epoch):

        """
        Set parameter max_iter in our model

        :param max_epoch: maximum number of epochs to do during training
        """

        self.model_frame.set_params(**{'max_iter': max_epoch})


class Cnn(Model, torch.nn.Module):
    def __init__(self, num_classes, activation='relu', lr=0.001, alpha=0.0, eps=1e-8, drop_rate=0.5, b_size=15,
                 num_epoch=10, valid_size=0.10, tol=0.005, num_stop_epoch=10, lr_decay_rate=5, num_lr_decay=3,
                 save_path=None):

        """
        Mother class for all cnn pytorch model. Only build layer and foward are not implemented in this model.

        :param num_classes: Number of class
        :param activation: Activation function (default: relu)
        :param lr: The initial learning rate used with the Adam Optimizer
        :param alpha: L2 penalty (regularization term) parameter as float (default: 0.0)
        :param eps: Adam optimizer hyper-parameters used to improve numerical stability (default: 1e-8)
        :param drop_rate: Dropout rate of each node of all fully connected layer (default: 0.5)
        :param b_size: Batch size as integer (default: 15)
        :param num_epoch: Number of epoch to do during the training (default: 10)
        :param valid_size: Portion of the data that will be used for validation.
        :param tol: Minimum difference between two epoch validation accuracy to consider that there is an improvement.
        :param num_stop_epoch: Number of consecutive epoch with no improvement on the validation accuracy
                               before early stopping
        :param lr_decay_rate: Rate of the learning rate decay when the optimizer does not seem to converge
        :param num_lr_decay: Number of learning rate decay step we do before stop training when the optimizer does not
                             seem to converge.
        :param save_path: Directory path where the checkpoints will be write during the training
        """

        Model.__init__(self, {"lr": Hyperparameter("lr", HPtype.real, [lr]),
                              "alpha": Hyperparameter("alpha", HPtype.real, [alpha]),
                              "eps": Hyperparameter("eps", HPtype.real, [eps]),
                              "dropout": Hyperparameter("dropout", HPtype.real, [drop_rate]),
                              "b_size": Hyperparameter("b_size", HPtype.integer, [b_size]),
                              "lr_decay_rate": Hyperparameter("lr_decay_rate", HPtype.integer, [lr_decay_rate]),
                              "activation": Hyperparameter("activation", HPtype.categorical, [activation])
                              })

        torch.nn.Module.__init__(self)

        # Base parameters (Parameters that will not change during training or hyperparameters search)
        self.classes = num_classes
        self.num_epoch = num_epoch
        self.device_ = torch.device("cpu")

        # early stopping parameters
        self.valid_size = valid_size
        self.tol = tol
        self.num_stop_epoch = num_stop_epoch
        self.num_lr_decay = num_lr_decay
        self.path = save_path

        # Hyperparameters dictionary
        self.hparams = {"lr": lr, "alpha": alpha, "eps": eps, "dropout": drop_rate, "b_size": b_size,
                        "lr_decay_rate": lr_decay_rate, "activation": activation}

        self.drop = torch.nn.Dropout(p=self.hparams["dropout"])
        self.soft = torch.nn.LogSoftmax(dim=-1)
        self.criterion = torch.nn.NLLLoss()

    @staticmethod
    def conv_out_size(in_size, conv_size, conv_type, pool):

        """
        Calculate the output resulting of a convolution layer and it corresponding pooling layer

        :param in_size: A numpy array of length 2 that represent the input image size. (height, width)
        :param conv_size: The convolutional kernel size as integer
        :param conv_type:  Convolution type: (0: Valid (no zero padding), 1: Same (zero padding added))
        :param pool: A numpy array of length 3 that represent pooling layer parameters. (type, height, width)
        :return: A numpy array of length 2 that represent the output image size. (height, width)
        """

        if conv_size == 0:
            raise Exception("Convolutional kernel of size 0")

        # In case of no padding out_size = in_size - (kernel_size - 1)
        if conv_type == 0:
            out_size = in_size - conv_size + 1
        else:
            out_size = in_size

        if np.any(pool[1:] == 0) & pool[0] != 0:
            raise Exception("Pooling kernel of size: {}, {}".format(pool[1], pool[2]))

        elif pool[0] == 1 or pool[0] == 2:
            out_size = np.floor([out_size[0] / pool[1], out_size[1] / pool[2]])

        elif pool[0] == 3 or pool[0] == 4:
            out_size = np.array([pool[1], pool[2]])

        return out_size.astype(int)

    @staticmethod
    def pad_size(conv_size, conv_type):

        """
        Compute the zero padding to add to each side of a dimension

        :param conv_size: Size of the kernel size as integer (Exemple: 3 for a kernel of 3x3)
        :param conv_type: Convolution type: (0: Valid (no zero padding), 1: Same (zero padding added))
        :return: Number of padding cells to add on each side of the image.
        """

        if conv_type == 1:
            return int((conv_size - 1) / 2)
        else:
            return 0

    @staticmethod
    def build_pooling_layer(pool_layer):

        """
        This method is used to generate a pooling layer to build the convolutional layer

        :param pool_layer: A numpy array of length 3 that represent pooling layer parameters. (type, height, width)
                        (type: 0: No pooling; 1: MaxPool; 2: AvgPool; 3: Adaptative MaxPool; 4: Adaptative AvgPool)
        :return: A torch.nn module that correspond to the given pooling specification
        """

        if pool_layer[0] == 1:
            return torch.nn.MaxPool2d(kernel_size=(pool_layer[1], pool_layer[2]))

        elif pool_layer[0] == 2:
            return torch.nn.AvgPool2d(kernel_size=(pool_layer[1], pool_layer[2]))

        elif pool_layer[0] == 3:
            return torch.nn.AdaptiveMaxPool2d(output_size=(pool_layer[1], pool_layer[2]))

        elif pool_layer[0] == 4:
            return torch.nn.AdaptiveAvgPool2d(output_size=(pool_layer[1], pool_layer[2]))

        elif pool_layer[0] > 4:
            raise Exception("pooling type {} does not correspond to a correct type of pooling layer. Choose between"
                            "\n 0: No pooling; 1: MaxPool; 2: AvgPool; 3: Adaptative MaxPool; 4: Adaptative AvgPool")
        else:
            return None

    def set_max_epoch(self, max_epoch):

        """
        Set parameter num_epoch in our model

        :param max_epoch: maximum number of epochs to do during training
        """

        self.num_epoch = max_epoch

    def get_activation_function(self):

        """
        This method is used to generate an activation function to build the CNN layer.

        :return: A torch.nn module that correspond to the activation function
        """

        if self.hparams["activation"] == "relu":
            return torch.nn.ReLU()
        elif self.hparams["activation"] == "elu":
            return torch.nn.ELU()
        elif self.hparams["activation"] == "prelu":
            return torch.nn.PReLU()
        elif self.hparams["activation"] == "sigmoid":
            return torch.nn.Sigmoid()
        elif self.hparams["activation"] == "swish":
            return Swish()
        elif self.hparams["activation"] == "mish":
            return Mish()
        else:
            raise Exception("No such activation has this name: {}".format(self.hparams["activation"]))

    def set_hyperparameters(self, hyperparams):

        """
        Function that set the new hyperparameters

        :param hyperparams: Dictionary specifing hyper-parameters to change.
        """

        for hp in hyperparams:
            if hp in self.hparams:
                self.hparams[hp] = hyperparams[hp]
            else:
                raise Exception('No such hyper-parameter "{}" in our model'.format(hp))

        self.drop.p = self.hparams["dropout"]

    def init_weights(self, m):

        """
        Initialize the weights of the fully connected layer and convolutional layer with Xavier normal initialization
        and Kamming normal initialization respectively.

        :param m: A torch.nn module of the current model. If this module is a layer, then we initialize its weights.
        """

        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

        elif type(m) == torch.nn.Conv2d:

            if self.hparams["activation"] == "relu" and self.hparams["activation"] == "sigmoide":
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity=self.hparams["activation"])
            else:
                torch.nn.init.kaiming_normal_(m.weight)
            if not(m.bias is None):
                torch.nn.init.zeros_(m.bias)

        elif type(m) == torch.nn.BatchNorm2d:
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def print_params(self):

        """
        Print all weight of the model in the terminal
        """

        for param in self.parameters():
            print(param.name, param.data)

    def save_checkpoint(self, epoch, loss, accuracy):

        """
        Save the model and his at a the current state if the self.path is not None.

        :param epoch: Current epoch of the training
        :param loss: Current loss of the training
        :param accuracy: Current validation accuracy
        """

        if self.path is not None:
            torch.save({"epoch": epoch,
                        "model_state_dict": self.state_dict(),
                        "loss": loss,
                        "accuracy": accuracy}, self.path)

    def restore(self):

        """
        Restore the weight from the last checkpoint saved during training
        """

        if self.path is not None:
            checkpoint = torch.load(self.path)
            self.load_state_dict(checkpoint['model_state_dict'])

    def switch_device(self, _device):

        """
        Switch the used device that our model will use for training and prediction

        :param _device: The device name (cpu, gpu) as string
        """

        if _device == "gpu":
            self.device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device_ = torch.device("cpu")

        self.to(self.device_)

    def set_train_valid_loader(self, X_train=None, t_train=None, dtset=None):

        """
        Split a torch dataset or features and labels numpy arrays into two dataset or two features and two labels numpy
        arrays respectively. Finally transform them into data loaders for training and validation

        :param X_train: NxD numpy array of the observations of the training set
        :param t_train: Nx1 numpy array classes associated with each observations in the training set
        :param dtset: A torch dataset which contain our train data points and labels
        :return: A data loarder for training and a data loader for validation
        """

        if dtset is None:
            # x_train, t_train, x_valid, t_valid
            x_t, t_t, x_v, t_v = validation_split(features=X_train, labels=t_train, valid_size=self.valid_size)

            train_loader = create_dataloader(x_t, t_t, self.hparams["b_size"], shuffle=True)
            valid_loader = create_dataloader(x_v, t_v, self.hparams["b_size"], shuffle=False)

        else:
            train_set, valid_set = validation_split(dtset=dtset, valid_size=self.valid_size)

            train_loader = dataset_to_loader(train_set, self.hparams["b_size"], shuffle=True)
            valid_loader = dataset_to_loader(valid_set, self.hparams["b_size"], shuffle=False)

        return train_loader, valid_loader

    def standard_epoch(self, train_loader, optimizer):
        """
        Make a standard training epoch

        :param train_loader: A torch data_loader that contain the features and the labels for training.
        :param optimizer: The torch optimizer that will used to train the model.
        :return: The average training loss
        """

        sum_loss = 0
        it = 0

        for step, data in enumerate(train_loader, 0):
            features, labels = data[0].to(self.device_), data[1].to(self.device_)

            optimizer.zero_grad()

            # training step
            pred = self(features)
            loss = self.criterion(pred, labels)
            loss.backward()
            optimizer.step()

            # Save the loss
            sum_loss += loss
            it += 1

        return sum_loss.item() / it

    def fit(self, X_train=None, t_train=None, dtset=None, verbose=False, gpu=True):

        """
        Train our model

        :param X_train: NxD numpy array of the observations of the training set
        :param t_train: Nx1 numpy array classes associated with each observations in the training set
        :param dtset: A torch dataset which contain our train data points and labels
        :param verbose: print the loss during training
        :param gpu: True: Train the model on the gpu. False: Train the model on the cpu
        """
        
        train_loader, valid_loader = self.set_train_valid_loader(X_train, t_train, dtset)

        # Indicator for early stopping
        best_accuracy = 0
        best_loss = float("inf")
        last_saved_loss = float("inf")
        best_epoch = -1
        num_epoch_no_change = 0
        lr_decay_step = 0
        learning_rate = self.hparams["lr"]

        # Go in training to activate dropout
        self.train()

        self.apply(self.init_weights)

        if gpu:
            self.switch_device("gpu")

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=self.hparams["alpha"],
                                     eps=self.hparams["eps"], amsgrad=False)

        with tqdm(total=self.num_epoch) as t:
            for epoch in range(self.num_epoch):
                # ------------------------------------------------------------------------------------------
                #                                       TRAINING PART
                # ------------------------------------------------------------------------------------------

                training_loss = self.standard_epoch(train_loader, optimizer)
                current_accuracy = self.accuracy(dt_loader=valid_loader)

                # ------------------------------------------------------------------------------------------
                #                                   EARLY STOPPING PART
                # ------------------------------------------------------------------------------------------
                if training_loss < last_saved_loss and current_accuracy >= best_accuracy:
                    self.save_checkpoint(epoch, training_loss, current_accuracy)
                    best_accuracy = current_accuracy
                    last_saved_loss = training_loss

                if self.tol < 1 - (training_loss / best_loss):
                    best_loss = training_loss
                    best_epoch = epoch
                    num_epoch_no_change = 0

                elif num_epoch_no_change < self.num_stop_epoch - 1:
                    num_epoch_no_change += 1

                # Learning rate decay step
                elif lr_decay_step < self.num_lr_decay:

                    # We update the learning rate and restart the optimizer
                    learning_rate /= self.hparams["lr_decay_rate"]
                    optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate,
                                                 weight_decay=self.hparams["alpha"],
                                                 eps=self.hparams["eps"], amsgrad=False)
                    lr_decay_step += 1
                    num_epoch_no_change = 0

                else:
                    break

                if verbose:
                    t.postfix = "avg loss: {:.4f}, validation: {:.2f}%, best accuracy: {:.2f}%, " \
                                "best epoch: {}, learning rate: {:.8f}".format(
                                 training_loss, current_accuracy * 100, best_accuracy * 100, best_epoch + 1,
                                 learning_rate)
                t.update()
            # We restore the weight of the model at his best epoch
            self.restore()

    def predict(self, X):

        """
        Predict classes for our observations in the input array X

        :param X: NxD numpy array of observations {N : nb of obs, D : nb of dimensions}
        :return: Nx1 numpy array of classes predicted for each observation
        """

        with torch.no_grad():
            out = torch.Tensor.cpu(self(X)).numpy()
        return np.argmax(out, axis=1)

    def accuracy(self, dt_loader):

        """
        Compute the accuracy of the model on a given data loader

        :param dt_loader: A torch data loader that contain test or validation data
        :return: The accuracy of the model
        """

        accuracy = np.array([])
        for data in dt_loader:
            features, labels = data[0].to(self.device_), data[1].numpy()
            pred = self.predict(features)

            accuracy = np.append(accuracy, np.where(pred == labels, 1, 0).mean())

        return accuracy.mean()

    def score(self, X=None, t=None, dtset=None):

        """
        Compute the accuracy of the model on a given test dataset

        :param X: NxD numpy array of observations {N : nb of obs, D : nb of dimensions}
        :param t: Nx1 numpy array of classes associated with each observation
        :param dtset: A torch dataset which contain our test data points and labels
        :return: The accuracy of the model.
        """

        if dtset is None:
            if X is None or t is None:
                raise Exception("Features or labels missing. X is None: {}, t is None: {}, dtset is None: {}".format(
                    X is None, t is None, dtset is None))
            else:
                test_loader = create_dataloader(X, t, self.hparams["b_size"], shuffle=False)
        else:
            test_loader = dataset_to_loader(dtset, self.hparams["b_size"], shuffle=False)

        self.eval()

        return self.accuracy(dt_loader=test_loader)


class CnnVanilla(Cnn):
    def __init__(self, num_classes, conv_config, pool_config, fc_config, activation='relu', input_dim=None, lr=0.001,
                 alpha=0.0, eps=1e-8, drop_rate=0.5, b_size=15, num_epoch=10, valid_size=0.10, tol=0.005,
                 num_stop_epoch=10, lr_decay_rate=5, num_lr_decay=3, save_path=None):

        """
        Class that generate a convolutional neural network using the sequential module of the Pytorch library.

        :param num_classes: Number of class
        :param conv_config: A Cx3 numpy matrix where each row represent the parameters of a 2D convolutional layer.
                           [i, 0]: Number of output channels of the ith layer
                           [i, 1]: Square convolution dimension of the ith layer
                           [i, 2]: Convolution type: (0: Valid (no zero padding), 1: Same (zero padding added))
        :param pool_config: An Cx3 numpy matrix where each row represent the parameters of a 2D pooling layer.
                          [i, 0]: Pooling layer type: 0: No pooling, 1: Max pooling, 2: Average pooling
                          [i, 1]: Pooling kernel height
                          [i, 2]: Pooling kernel width
        :param fc_config: A numpy array where each elements represent the number of nodes of a fully connected layer
        :param input_dim: Image input dimensions [height, width, deep]
        :param activation: Activation function (default: relu)
        :param lr: The initial learning rate used with the Adam Optimizer
        :param alpha: L2 penalty (regularization term) parameter as float (default: 0.0)
        :param eps: Adam optimizer hyper-parameters used to improve numerical stability (default: 1e-8)
        :param drop_rate: Dropout rate of each node of all fully connected layer (default: 0.5)
        :param b_size: Batch size as integer (default: 15)
        :param num_epoch: Number of epoch to do during the training (default: 10)
        :param valid_size: Portion of the data that will be used for validation.
        :param tol: Minimum difference between two epoch validation accuracy to consider that there is an improvement.
        :param num_stop_epoch: Number of consecutive epoch with no improvement on the validation accuracy
                               before early stopping
        :param lr_decay_rate: Rate of the learning rate decay when the optimizer does not seem to converge
        :param num_lr_decay: Number of learning rate decay step we do before stop training when the optimizer does not
                             seem to converge.
        :param save_path: Directory path where the checkpoints will be write during the training
        """

        Cnn.__init__(self, num_classes, activation=activation, lr=lr, alpha=alpha, eps=eps, drop_rate=drop_rate,
                     b_size=b_size, num_epoch=num_epoch, valid_size=valid_size, tol=tol, num_stop_epoch=num_stop_epoch,
                     lr_decay_rate=lr_decay_rate, num_lr_decay=num_lr_decay, save_path=save_path)

        # We need to save the model configuration parameters to rebuild it during the hyper-parameters research
        self.conv_config = conv_config
        self.pool_config = pool_config
        self.fc_config = fc_config

        # We need a special type of list to ensure that torch detect every layer and node of the neural net
        self.conv = None
        self.fc = None
        self.num_flat_features = 0

        # Default image dimension. Height: 28, width: 28 and deep: 1 (MNIST)
        if input_dim is None:
            self.input_dim = np.array([28, 28, 1])
        else:
            self.input_dim = input_dim

        # We build the model
        self.build_layer(self.conv_config, self.pool_config, self.fc_config, self.input_dim)

    def set_hyperparameters(self, hyperparams):

        """
        Function that set the new hyper-parameters and rebuild the model after the update

        :param hyperparams: Dictionary specifying hyper-parameters to change.
        """

        Cnn.set_hyperparameters(self, hyperparams)

        self.conv = self.fc = None
        self.num_flat_features = 0

        self.build_layer(self.conv_config, self.pool_config, self.fc_config, self.input_dim)

    def build_layer(self, conv_layer, pool_list, fc_nodes, input_dim):

        """
        Create the model architecture

        :param conv_layer: A Cx3 numpy matrix where each row represent the parameters of a 2D convolutional layer.
                           [i, 0]: Number of output channels of the ith layer
                           [i, 1]: Square convolution dimension of the ith layer
                           [i, 2]: Convolution type: (0: Valid (no zero padding), 1: Same (zero padding added))
        :param pool_list: An Cx3 numpy matrix where each row represent the parameters of a 2D pooling layer.
                          [i, 0]: Pooling layer type: 0: No pooling, 1: Max pooling, 2: Average pooling
                          [i, 1]: Pooling kernel height
                          [i, 2]: Pooling kernel width
        :param fc_nodes: A numpy array where each elements represent the number of nodes of a fully connected layer
        :param input_dim: Image input dimensions [height, width, deep]
        """

        # ------------------------------------------------------------------------------------------
        #                                   CONVOLUTIONAL PART
        # ------------------------------------------------------------------------------------------
        # First convolutional layer
        conv_list = [torch.nn.Conv2d(input_dim[2], conv_layer[0, 0], conv_layer[0, 1],
                                     padding=self.pad_size(conv_layer[0, 1], conv_layer[0, 2])),
                     self.get_activation_function()]
        # Pooling
        if pool_list[0, 0] != 0:
            conv_list.extend([self.build_pooling_layer(pool_list[0])])

        # We need to compute the input size of the fully connected layer
        size = self.conv_out_size(input_dim[0:2], conv_layer[0, 1], conv_layer[0, 2], pool_list[0])

        # All others convolutional layers
        for it in range(1, len(conv_layer)):
            # Convolution
            conv_list.extend([torch.nn.Conv2d(conv_layer[it - 1, 0], conv_layer[it, 0], conv_layer[it, 1],
                                              padding=self.pad_size(conv_layer[it, 1], conv_layer[it, 2])),
                              self.get_activation_function()])
            # Pooling
            if pool_list[it, 0] != 0:
                conv_list.extend([self.build_pooling_layer(pool_list[it])])

            # Update the output size
            size = self.conv_out_size(size, conv_layer[it, 1], conv_layer[it, 2], pool_list[it])

        # We create the sequential of the convolutional network part
        self.conv = torch.nn.Sequential(*conv_list)

        # ------------------------------------------------------------------------------------------
        #                                   FULLY CONNECTED PART
        # ------------------------------------------------------------------------------------------
        # Compute the fully connected input layer size
        self.num_flat_features = size[0] * size[1] * conv_layer[-1, 0]

        # First fully connected layer
        fc_list = [torch.nn.Linear(self.num_flat_features, fc_nodes[0]), self.get_activation_function(), self.drop]

        # All other fully connected layer
        for it in range(1, len(fc_nodes)):
            fc_list.extend([torch.nn.Linear(fc_nodes[it - 1], fc_nodes[it]), self.get_activation_function(), self.drop])

        # Output layer
        fc_list.extend([torch.nn.Linear(fc_nodes[-1], self.classes), self.soft])

        self.fc = torch.nn.Sequential(*fc_list)

    def forward(self, x):

        """
        Define the forward pass of the neural network

        :param x: Input tensor of size BxD where B is the Batch size and D is the features dimension
        :return: Output tensor of size num_class x 1.
        """
        conv_out = self.conv(x)
        output = self.fc(conv_out.view(-1, self.num_flat_features))
        return output


class ResNet(Cnn):
    def __init__(self, num_classes, conv_config, res_config, pool1, pool2, fc_config, activation='relu', version=1,
                 input_dim=None, lr=0.001, alpha=0.0, eps=1e-8, drop_rate=0.0, b_size=15, num_epoch=10, valid_size=0.10,
                 tol=0.005, num_stop_epoch=10, lr_decay_rate=5, num_lr_decay=3, save_path=None):

        """
        Class that generate a ResNet neural network inpired by the model from the paper "Deep Residual Learning for
        Image Recogniton" (Ref 1).

        :param num_classes: Number of classes
        :param conv_config: A tuple that represent the parameters of the first convolutional layer.
                     [0]: Number of output channels (features maps)
                     [1]: Kernel size: (Example: 3.  For a 3x3 kernel)
                     [2]: Convolution type: (0: Valid (no zero padding), 1: Same (zero padding added))
        :param res_config: A Cx2 numpy matrix where each row represent the parameters of a sub-sampling level.
                           [i, 0]: Number of residual modules
                           [i, 2]: Kernel size of the convolutional layers
        :param pool1: A tuple that represent the parameters of the pooling layer that came after the first conv layer
        :param pool2: A tuple that represent the parameters of the last pooling layer before the fully-connected layers
                      [0]: Pooling layer type: 0: No pooling, 1: Max pooling, 2: Average pooling
                      [1]: Pooling kernel height
                      [2]: Pooling kernel width
        :param fc_config: A numpy array where each elements represent the number of nodes of a fully connected layer
        :param input_dim: Image input dimensions [height, width, deep]
        :param activation: Activation function (default: relu)
        :param version: Which version of the ResNet should be use. V1: Post activation, V2: Pre activation
        :param lr: The initial learning rate used with the Adam Optimizer
        :param alpha: L2 penalty (regularization term) parameter as float (default: 0.0)
        :param eps: Adam optimizer hyper-parameters used to improve numerical stability (default: 1e-8)
        :param drop_rate: Dropout rate of each node of all fully connected layer (default: 0.0)
        :param b_size: Batch size as integer (default: 15)
        :param num_epoch: Number of epoch to do during the training (default: 10)
        :param valid_size: Portion of the data that will be used for validation.
        :param tol: Minimum difference between two epoch validation accuracy to consider that there is an improvement.
        :param num_stop_epoch: Number of consecutive epoch with no improvement on the validation accuracy
                               before early stopping
        :param lr_decay_rate: Rate of the learning rate decay when the optimizer does not seem to converge
        :param num_lr_decay: Number of learning rate decay step we do before stop training when the optimizer does not
                             seem to converge.
        :param save_path: Directory path where the checkpoints will be write during the training
        """

        Cnn.__init__(self, num_classes, activation=activation, lr=lr, alpha=alpha, eps=eps, drop_rate=drop_rate,
                     b_size=b_size, num_epoch=num_epoch, valid_size=valid_size, tol=tol, num_stop_epoch=num_stop_epoch,
                     lr_decay_rate=lr_decay_rate, num_lr_decay=num_lr_decay, save_path=save_path)

        # Hyper-parameters specifics to the ResNet
        if version == 1 or version == 2:
            self.hparams['version'] = version
            self.HP_space["version"] = Hyperparameter("version", HPtype.integer, [version])
        else:
            raise Exception("Version parameter set to {}. Choose 1 or 2".format(version))

        # We need to save the model configuration parameters to rebuild it during the hyper-parameters research
        self.conv_config = conv_config
        self.res_config = res_config
        self.pool = [pool1, pool2]
        self.fc_config = fc_config

        # We need a special type of list to ensure that torch detect every layer and node of the neural net
        self.conv = None
        self.fc = None
        self.num_flat_features = 0
        self.out_layer = None

        # Default image dimension. Height: 32, width: 32 and deep: 3 (CIFAR10)
        if input_dim is None:
            self.input_dim = np.array([32, 32, 3])
        else:
            self.input_dim = input_dim

        self.build_layer(self.conv_config, self.res_config, self.pool[0], self.pool[1], self.fc_config, self.input_dim)

    def set_hyperparameters(self, hyperparams):

        """
        Function that set the new hyper-parameters and rebuild the model after the update

        :param hyperparams: Dictionary specifying hyper-parameters to change.
        """

        Cnn.set_hyperparameters(self, hyperparams)

        self.conv = self.fc = self.out_layer = None
        self.num_flat_features = 0

        self.build_layer(self.conv_config, self.res_config, self.pool[0], self.pool[1], self.fc_config, self.input_dim)

    def build_layer(self, conv, res_config, pool1, pool2, fc_nodes, input_dim):

        """
        Create the model architecture

        :param conv: A tuple that represent the parameters of the first convolutional layer.
                     [0]: Number of output channels (features maps)
                     [1]: Kernel size: (Example: 3.  For a 3x3 kernel)
                     [2]: Convolution type: (0: Valid (no zero padding), 1: Same (zero padding added))
        :param res_config:A Cx2 numpy matrix where each row represent the parameters of a sub-sampling level.
                          [i, 0]: Number of residual modules
                          [i, 1]: Kernel size of the convolutional layers
        :param pool1: A tuple that represent the parameters of the pooling layer that came after the first conv layer
        :param pool2: A tuple that represent the parameters of the last pooling layer before the fully-connected layers
                      [0]: Pooling layer type: 0: No pooling, 1: Max pooling, 2: Average pooling
                      [1]: Pooling kernel height
                      [2]: Pooling kernel width
        :param fc_nodes: A numpy array where each elements represent the number of nodes of a fully connected layer
        :param input_dim: Image input dimensions [height, width, deep]
        """

        # ------------------------------------------------------------------------------------------
        #                                   CONVOLUTIONAL PART
        # ------------------------------------------------------------------------------------------
        # First convolutional layer
        conv_list = [torch.nn.Conv2d(input_dim[2], conv[0], conv[1], padding=self.pad_size(conv[1], conv[2]))]

        # No batch norm and activation if its a pre-activation ResNet
        if self.hparams['version'] == 1:
            conv_list.extend([torch.nn.BatchNorm2d(conv[0]),
                              self.get_activation_function()])

        if pool1[0] != 0:
            conv_list.extend([self.build_pooling_layer(pool1)])

        # We need to compute the input size of the fully connected layer
        size = self.conv_out_size(input_dim[0:2], conv[1], conv[2], pool1)

        # ------------------------------------------------------------------------------------------
        #                                      RESIDUAL PART
        # ------------------------------------------------------------------------------------------

        # We select the right resnet module according to the given version
        if self.hparams['version'] == 1:
            res_module = ResModuleV1
        else:
            res_module = ResModuleV2

        # Number of features that enter in the first residual module
        f_in = conv[0]

        # Construct the chain of residual module
        for it in range(len(res_config)):
            conv_list.extend([res_module(f_in, res_config[it, 1], self.hparams["activation"],
                                         twice=(it != 0), subsample=(it != 0))])

            subsample = False

            # Update features maps information
            if it > 0:
                f_in *= 2
                size = size / 2
                if self.hparams['version'] == 2:
                    subsample = True

            for _ in range(res_config[it, 0] - 1):
                conv_list.extend([res_module(f_in, res_config[it, 1], self.hparams["activation"],
                                             twice=False, subsample=subsample)])

        if pool2[0] != 0:
            conv_list.extend([self.build_pooling_layer(pool2)])

        # We need to compute the input size of the fully connected layer
        size = self.conv_out_size(size, res_config[-1, 1], 2, pool2)

        self.conv = torch.nn.Sequential(*conv_list)

        # ------------------------------------------------------------------------------------------
        #                                   FULLY CONNECTED PART
        # ------------------------------------------------------------------------------------------
        # Compute the fully connected input layer size
        self.num_flat_features = size[0] * size[1] * f_in
        fc_list = []

        if fc_nodes is None:
            num_last_nodes = self.num_flat_features

        else:
            # First fully connected layer
            fc_list.extend([torch.nn.Linear(self.num_flat_features, fc_nodes[0]),
                            self.get_activation_function()])

            if self.hparams['dropout'] > 0:
                fc_list.extend([self.drop])

            # All others hidden layers
            for it in range(1, len(fc_nodes)):
                fc_list.extend([torch.nn.Linear(fc_nodes[it - 1], fc_nodes[it]),
                                self.get_activation_function()])

                if self.hparams['dropout'] > 0:
                    fc_list.extend([self.drop])

            num_last_nodes = fc_nodes[-1]

        # Output layer
        fc_list.extend([torch.nn.Linear(num_last_nodes, self.classes), self.soft])

        self.fc = torch.nn.Sequential(*fc_list)

    def forward(self, x):

        """
        Define the forward pass of the neural network

        :param x: Input tensor of size BxD where B is the Batch size and D is the features dimension
        :return: Output tensor of size num_class x 1.
        """

        conv_out = self.conv(x)
        output = self.fc(conv_out.view(-1, self.num_flat_features))
        return output


class SimpleResNet(ResNet):
    def __init__(self, num_classes, num_res, activation='relu', version=1, input_dim=None, lr=0.001, alpha=0.0,
                 eps=1e-8, b_size=15, num_epoch=10, valid_size=0.10, tol=0.005, num_stop_epoch=10, lr_decay_rate=5,
                 num_lr_decay=3, save_path=None):

        """
        Class that generate a ResNet neural network inpired by the model from the paper "Deep Residual Learning for
        Image Recogniton" (Ref 1).

        :param num_classes: Number of classes
        :param num_res: Number of residual module in each sub sampling level
        :param input_dim: Image input dimensions [height, width, deep]
        :param activation: Activation function (default: relu)
        :param version: Which version of the ResNet should be use. V1: Post activation, V2: Pre activation
        :param lr: The initial learning rate used with the Adam Optimizer
        :param alpha: L2 penalty (regularization term) parameter as float (default: 0.0)
        :param eps: Adam optimizer hyper-parameters used to improve numerical stability (default: 1e-8)
        :param b_size: Batch size as integer (default: 15)
        :param num_epoch: Number of epoch to do during the training (default: 10)
        :param valid_size: Portion of the data that will be used for validation.
        :param tol: Minimum difference between two epoch validation accuracy to consider that there is an improvement.
        :param num_stop_epoch: Number of consecutive epoch with no improvement on the validation accuracy
                               before early stopping
        :param lr_decay_rate: Rate of the learning rate decay when the optimizer does not seem to converge
        :param num_lr_decay: Number of learning rate decay step we do before stop training when the optimizer does not
                             seem to converge.
        :param save_path: Directory path where the checkpoints will be write during the training
        """

        conv = np.array([16, 3, 1])  # First conv layer: 16 output channels, kernel 3x3 and same padding type
        res = np.array([[num_res, 3], [num_res, 3], [num_res, 3]])
        pool1 = np.array([0, 0, 0])  # No pooling layer after the first convolution
        pool2 = np.array([4, 1, 1])  # Adaptive average pooling after the last convolution layer.
        fc_config = None  # No extra fully connected after the the convolutional part.

        ResNet.__init__(self, num_classes=num_classes, conv_config=conv, res_config=res, pool1=pool1, pool2=pool2,
                        fc_config=fc_config, activation=activation, version=version, input_dim=input_dim, lr=lr,
                        alpha=alpha, eps=eps, b_size=b_size, num_epoch=num_epoch, num_stop_epoch=num_stop_epoch,
                        lr_decay_rate=lr_decay_rate, num_lr_decay=num_lr_decay, valid_size=valid_size, tol=tol,
                        save_path=save_path)

        self.hparams['num_res'] = num_res
        self.HP_space["num_res"] = Hyperparameter("num_res", HPtype.integer, [num_res])

    def set_hyperparameters(self, hyperparams):

        """
        Function that set the new hyper-parameters and rebuild the model after the update

        :param hyperparams: Dictionary specifying hyper-parameters to change.
        """

        Cnn.set_hyperparameters(self, hyperparams)

        # We update the residual layer configuration.
        self.res_config = np.array([[self.hparams['num_res'], 3] for _ in range(3)])

        self.conv = self.fc = self.out_layer = None
        self.num_flat_features = 0

        self.build_layer(self.conv_config, self.res_config, self.pool[0], self.pool[1], self.fc_config, self.input_dim)
