import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):

    def __init__(self, input_features, outputs):
        super(DQNNetwork, self).__init__()
        self.input = nn.Linear(input_features, 512)
        self.hidden = nn.Linear(512, 128)
        self.head = nn.Linear(128, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden(x))
        return self.head(x)
