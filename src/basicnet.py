import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, idim, hdim=[], odim=1):
        super(Net, self).__init__()
        self._nhlayers = len(hdim)
        assert self._nhlayers >= 1

        self._af = nn.ReLU
        self._of = nn.Sigmoid
        self._l_dim = [idim] + [k for k in hdim] + [odim]

        layers = []
        for i in range(self._nhlayers):
            layers.append(nn.Linear(self._l_dim[i], self._l_dim[i+1]))
            layers.append(self._af())
            layers.append(nn.BatchNorm1d(self._l_dim[i+1]))
        layers += [nn.Linear(hdim[-1], odim), self._of()]

        self._layers = nn.Sequential(*layers)

    def forward(self, x):
        return self._layers(x)

    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().numpy()

    def get_embedding(self, x, k):
        assert k in list(range(1, self._nhlayers+1))
        for i in range(2*k): # include activation
            x = self._layers[i](x)
        return x
