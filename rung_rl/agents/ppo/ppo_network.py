import torch
import torch.nn as nn


# import torch.nn.functional as F


class PPONetwork(nn.Module):

    def __init__(self, input_features, hidden_size, outputs):
        super(PPONetwork, self).__init__()
        self.input = nn.Linear(input_features, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, outputs)
        # self.hidden = nn.Linear(128, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = torch.tanh(self.input(x))

        x = torch.tanh(self.hidden(x))

        logits = self.output(x)

        return logits
