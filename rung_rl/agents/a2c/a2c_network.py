import torch.nn as nn
import torch
# import torch.nn.functional as F


class A2CNetwork(nn.Module):

    def __init__(self, input_features, hidden_neurons, outputs):
        super(A2CNetwork, self).__init__()
        self.input = nn.Linear(input_features, hidden_neurons)
        self.actor_hidden = nn.Linear(hidden_neurons, hidden_neurons)
        self.actor = nn.Linear(hidden_neurons, outputs)
        self.critic_hidden = nn.Linear(hidden_neurons, hidden_neurons)
        self.critic = nn.Linear(hidden_neurons, 1)
        # self.hidden = nn.Linear(128, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        init = torch.tanh(self.input(x))
        
        logits = torch.tanh(self.actor_hidden(init))
        logits = self.actor(logits)

        value = torch.tanh(self.critic_hidden(init))
        value = self.critic(value)
        
        
        return logits, value
