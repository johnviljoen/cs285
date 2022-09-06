from typing import Union

import torch
from torch import nn

Activation = Union[str, nn.Module]


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            MLP (nn.Module)
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    # TODO: return a MLP. This should be an instance of nn.Module
    # Note: nn.Sequential is an instance of nn.Module.

    # with 2 hidden layers, we have an input layer, 2 hidden layers, and an output layer,
    # all of which are fully connected

    class NN(torch.nn.Module):
        def __init__(self,
         input_size=input_size, 
         output_size=output_size,
         n_layers=n_layers,
         size=size,
         activation=activation,
         output_activation=output_activation):

            # ignoring dtype and device for now.

            super(NN, self).__init__()
            self.n_layers = n_layers
            self.activation = activation
            self.output_activation = output_activation

            # form an MLP with correct input_size, output_size, and internal size
            self.fc = {}
            for i in range(n_layers+2):
                if i == 0:
                    self.fc[f'{i}'] = nn.Linear(input_size, size)
                elif i == n_layers+1:
                    self.fc[f'{i}'] = nn.Linear(size, output_size)
                else:
                    self.fc[f'{i}'] = torch.nn.Linear(size, size)

        def __call__(self, x):
            return self.forward(x)

        def forward(self, x):
            for i in range(self.n_layers+2):
                if i != n_layers+1:
                    x = self.activation(self.fc[f'{i}'](x))
                else: 
                    x = self.output_activation(self.fc[f'{i}'])(x)
            return x
    
    #class NN2(torch.nn.Module):
    #    def __init__(self,
    #     input_size=input_size, 
    #     output_size=output_size,
    #     n_layers=n_layers,
    #     size=size,
    #     activation=activation,
    #     output_activation=output_activation):

    #        # ignoring dtype and device for now.

    #        super(NN2, self).__init__()
    #        self.n_layers = n_layers
    #        self.activation = activation
    #        self.output_activation = output_activation

    #        self.fc1 = nn.Linear(input_size, size)
    #        self.fc2 = nn.Linear(size, output_size)

    #    def forward(self, x):
    #        x = self.activation(self.fc1(x))
    #        x = self.activation(self.fc2(x))
    #        x = self.output_activation(x)

    #        return x


    return NN()


device = None

def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
