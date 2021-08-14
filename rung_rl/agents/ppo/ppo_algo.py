import torch
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class PPO():
    def __init__(self,
                 actor,
                 critic,
                 actor_optimizer,
                 critic_optimizer,
                 clip_param=0.2,
                 ppo_epoch=4,
                 num_mini_batch=64,
                 value_loss_coef=1,
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 eps=None,
                 use_gae=False,
                 use_clipped_value_loss=False):

        self.actor = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.use_gae = use_gae
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def evaluate(self, states, actions):
        """
        Evaluates states with the given actions to find the probability of
        selecting the actions batch under the current policy and their
        associated predicted values
        Usually called with a batch of states and actions
        """

        raw_probs = self.actor(states)
        values = self.critic(states)
        dist = torch.distributions.Categorical(logits=raw_probs)
        log_probs = dist.log_prob(actions)
        return log_probs, values, dist.entropy().mean()

    def sample(self, states, actions, log_probs, returns, batch_size):
        """
        Converts a sample of size batch_size to multiple mini batches of
        size num_M
        """
        states = torch.cat(states)
        actions = torch.cat(actions)
        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns)
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), self.num_mini_batch, drop_last=True)
        for indices in sampler:
            # print(indices)
            yield states[indices], actions[indices], log_probs[indices], returns[indices]

    def update_ppo(self, states_batch, actions_batch, log_probs_batch, returns_batch, batch_size):
        # the total observations that were created with the environment running
        print("Batch size: {}".format(batch_size))
        action_loss_total = 0
        value_loss_total = 0
        entropy_total = 0
        # update for ppo_epoch times
        for _ in range(self.ppo_epoch):

            # create small batches of transitions
            for sample in self.sample(states_batch, actions_batch, log_probs_batch,
                                      returns_batch, batch_size):
                states, actions, log_probs, returns = sample

                # Evaluating old actions and states under the new policy to find
                # the surrogate
                new_log_probs, values, entropy = self.evaluate(states, actions)
                values = torch.squeeze(values)
                returns = torch.squeeze(returns)

                ratios = torch.exp(new_log_probs - log_probs.detach())

                if self.use_gae:
                    advantages = returns
                else:
                    advantages = returns - values.detach()

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                entropy_loss = self.entropy_coef * entropy
                loss = actor_loss - entropy_loss
                self.actor_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # optimize critic
                mse = torch.nn.MSELoss()
                value_loss = self.value_loss_coef * mse(values, returns)
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                action_loss_total += actor_loss.item()
                value_loss_total += value_loss.item()
                entropy_total += entropy_loss.item()

        num_updates = self.ppo_epoch * (batch_size / self.num_mini_batch)
        action_loss_total /= num_updates
        value_loss_total /= num_updates
        entropy_total /= num_updates

        return action_loss_total, value_loss_total, entropy_total

    # def update(self, rollouts):
    #     advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
    #     advantages = (advantages - advantages.mean()) / (
    #         advantages.std() + 1e-5)

    #     value_loss_epoch = 0
    #     action_loss_epoch = 0
    #     dist_entropy_epoch = 0

    #     for e in range(self.ppo_epoch):
    #         if self.actor_critic.is_recurrent:
    #             data_generator = rollouts.recurrent_generator(
    #                 advantages, self.num_mini_batch)
    #         else:
    #             data_generator = rollouts.feed_forward_generator(
    #                 advantages, self.num_mini_batch)

    #         for sample in data_generator:
    #             obs_batch, recurrent_hidden_states_batch, actions_batch, \
    #                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
    #                     adv_targ = sample

    #             # Reshape to do in a single forward pass for all steps
    #             values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
    #                 obs_batch, recurrent_hidden_states_batch, masks_batch,
    #                 actions_batch)

    #             ratio = torch.exp(action_log_probs -
    #                               old_action_log_probs_batch)
    #             surr1 = ratio * adv_targ
    #             surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
    #                                 1.0 + self.clip_param) * adv_targ
    #             action_loss = -torch.min(surr1, surr2).mean()

    #             if self.use_clipped_value_loss:
    #                 value_pred_clipped = value_preds_batch + \
    #                     (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
    #                 value_losses = (values - return_batch).pow(2)
    #                 value_losses_clipped = (
    #                     value_pred_clipped - return_batch).pow(2)
    #                 value_loss = 0.5 * torch.max(value_losses,
    #                                              value_losses_clipped).mean()
    #             else:
    #                 value_loss = 0.5 * (return_batch - values).pow(2).mean()

    #             self.optimizer.zero_grad()
    #             (value_loss * self.value_loss_coef + action_loss -
    #              dist_entropy * self.entropy_coef).backward()
    #             nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
    #                                      self.max_grad_norm)
    #             self.optimizer.step()

    #             value_loss_epoch += value_loss.item()
    #             action_loss_epoch += action_loss.item()
    #             dist_entropy_epoch += dist_entropy.item()

    #     num_updates = self.ppo_epoch * self.num_mini_batch

    #     value_loss_epoch /= num_updates
    #     action_loss_epoch /= num_updates
    #     dist_entropy_epoch /= num_updates
    #     print(value_loss_epoch, action_loss_epoch, dist_entropy_epoch)
    #     return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
