import torch.nn as nn
import torch
import torch.nn.init as init
# import torch.nn.functional as F


class DQNNetwork(nn.Module):

    def __init__(self, input_features, outputs):
        super(DQNNetwork, self).__init__()
        self.input = nn.Linear(input_features, 128)
        self.hidden = nn.Linear(128, 64)
        self.head = nn.Linear(64, outputs)
        # self.hidden = nn.Linear(128, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.hidden(x))
        return self.head(x)

"""
The recurrent version of the q network. 
"""
class DRQNNetwork(nn.Module):

    def __init__(self, input_features, hidden_size, outputs):
        super(DRQNNetwork, self).__init__()
        self.input = nn.Linear(input_features, 512)
        self.gru = nn.GRU(512, hidden_size, 1)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, outputs)

        # init.xavier_normal(self.input.weight.data)
        # init.normal()
        # init(self.hidden.weight)
        # init(self.output.weight)

        # for param in self.gru.parameters():
            # if len(param.shape) >= 2:
                # init.orthogonal_(param.data)
            # else:
                # init.normal_(param.data)

    def forward(self, x, hidden_states):
        hidden_states = hidden_states.unsqueeze(0)
        x = torch.tanh(self.input(x))
        x = x.unsqueeze(0)
        out, hidden = self.gru(x, hidden_states)
        y = torch.tanh(self.hidden(out.squeeze(0)))
        return self.output(y), hidden.squeeze(0)
        
