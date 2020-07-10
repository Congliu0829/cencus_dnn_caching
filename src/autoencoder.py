import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, indim, outdim):
        super(AutoEncoder, self).__init__()
        self.indim = indim
        self.outdim = outdim

        self.hidden_e = nn.Sequential(nn.Linear(indim, int(indim/2)),
                                      nn.ReLU(),
                                      nn.Linear(int(indim/2), outdim))

        self.hidden_d = nn.Sequential(nn.Linear(outdim, int(indim/2)),
                                      nn.ReLU(),
                                      nn.Linear(int(indim/2), indim))

    def encode(self, x):
        return self.hidden_e(x)

    def decode(self, x):
        return self.hidden_d(x)

    def forward(self, x):
        return self.decode(self.encode(x))
