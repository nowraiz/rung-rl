import torch.nn as nn
import torch
# import torch.nn.functional as F


class NFSPNetwork(nn.Module):

    def __init__(self, input_features, outputs):
        super(NFSPNetwork, self).__init__()
        self.input = nn.Linear(input_features, 128)
        self.hidden = nn.Linear(128, 64)
        self.head = nn.Linear(64, outputs)
        # self.hidden = nn.Linear(128, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = torch.relu(self.input(x))
        x = torch.relu(self.hidden(x))
        return self.head(x)
