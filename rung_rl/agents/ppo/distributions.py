import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rung_rl.utils import AddBias, init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def renormalize(self, log, action_mask):
        action_mask_log = torch.tensor([0 if m == 1 else float("-inf") for m in action_mask], device=device)
        masked = log+action_mask_log
        return masked
        # log = F.softmax(log,dim=-1)

        # # print(log, action_mask)
        # mask = torch.tensor(action_mask, dtype=torch.bool).to(device)
        # actions = sum(mask).item() # valid actions
        # total_actions = len(log[0]) # total possible actions before
        # if actions == total_actions:
        #     return log
        # # masked_log = log.masked_select(mask)
        # negmask = ~mask
        # # print(negmask)
        # remaining_log = log.masked_select(negmask)
        # balance = sum(remaining_log).item()/(actions)
        # temp = log.tolist()
        # for i in range(len(temp[0])):
        #     if action_mask[i] == 0:
        #         temp[0][i] = 0
        #     else:
        #         temp[0][i] += balance
        # return torch.Tensor(temp).to(device)
        

    def forward(self, x,action_mask):
        x = self.linear(x)
        
        return FixedCategorical(logits=self.renormalize(x,action_mask))


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)