import torch.nn as nn
import torch
# import torch.nn.functional as F


class OracleNetwork(nn.Module):

    def __init__(self, input_features, outputs):
        super(OracleNetwork, self).__init__()
        self.input = nn.Linear(input_features, 256)
        self.hidden = nn.Linear(256, 256)
        self.head = nn.Linear(256, outputs)
        # self.hidden = nn.Linear(128, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.hidden(x))
        return self.head(x)
