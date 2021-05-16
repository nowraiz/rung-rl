import torch.nn as nn
import torch
# import torch.nn.functional as F


class PPONetwork(nn.Module):

    def __init__(self, input_features, alpha_size, actor_size, critic_size, outputs):
        super(PPONetwork, self).__init__()
        self.input = nn.Linear(input_features, alpha_size)
        self.actor_hidden = nn.Linear(alpha_size, actor_size)
        self.actor = nn.Linear(actor_size, outputs)
        self.critic_hidden = nn.Linear(alpha_size, critic_size)
        self.critic = nn.Linear(critic_size, 1)
        # self.hidden = nn.Linear(128, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        init = torch.relu(self.input(x))
        
        logits = torch.relu(self.actor_hidden(init))
        logits = self.actor(logits)

        value = torch.relu(self.critic_hidden(init))
        value = self.critic(value)
        
        
        return logits, value
