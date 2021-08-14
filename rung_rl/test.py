import torch
import torch.nn.functional as F

from rung_rl.agents.dqn.dqn_network import DQNNetwork

network = DQNNetwork(4, 2)
y = network(torch.tensor([
    [2, 6, 7, 8],
    [1, 2, 3, 4]], dtype=torch.float))
# y[0, 0] = float("-inf")
# action_mask = [1, 0]
# q = torch.tensor([0 if m == 1 else float("-inf") for m in action_mask])
# print(y)
# print(y+q)
y = F.softmax(y, dim=-1)
print(y)
# print(y.ma\

# print(y.max(1)[1].view(2, 1))
# print()
# print(F.softmax(y))
