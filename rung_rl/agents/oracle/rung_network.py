import torch.nn as nn
import torch
# import torch.nn.functional as F


class RungNetwork(nn.Module):

    def __init__(self, input_features, outputs):
        super(RungNetwork, self).__init__()
        self.input = nn.Linear(input_features, 32)
        self.hidden = nn.Linear(32, outputs)
        # self.head = nn.Linear(64, outputs)
        # self.hidden = nn.Linear(128, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = torch.relu(self.input(x))
        # x = torch.relu(self.hidden(x))
        return self.hidden(x)
