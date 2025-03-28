# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch

import verl
import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config):
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_reinforce_returns(token_level_rewards: torch.Tensor, eos_mask: torch.Tensor, gamma: float):
    """Compute returns for REINFORCE algorithm.

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL

    Returns:
        returns: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    with torch.no_grad():
        returns_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            if t == gen_len - 1:
                next_return = 0.0
            else:
                next_return = returns_reversed[0]

            current_return = token_level_rewards[:, t] + gamma * next_return
            returns_reversed.insert(0, current_return)

        returns = torch.stack(returns_reversed, dim=1)

        # In REINFORCE, advantages are just returns (no baseline subtraction)
        advantages = returns.clone()
        # Optional: normalize advantages
        advantages = verl_F.masked_whiten(advantages, eos_mask)

    return advantages, returns


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


def compute_gae_value_returns(data: verl.DataProto, eos_mask: torch.Tensor, n_samples, config, lam, gamma):
    # use implicit prm as value model. this algorithm can be converted to a special gae estimator
    # reward coefficient take no effect here.
    # gamma is always 1.0
    # update: r_ov can be removed to prevent last-token bias.
    with torch.no_grad():
        token_level_rewards = data.batch['gt_scores']
        q = data.batch['rm_scores'].clone()  # beta * logprob
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]
        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(-1)
        # make q equal value: for t at eos, q equals reward-sum(q). for later t, q is zero
        for i in range(q.shape[0]):
            if config.reward_model.prime_use_gt:
                q[i, valid_response_length[i] -
                  1] = token_level_rewards[i, valid_response_length[i] - 1] - q[i, :valid_response_length[i] - 1].sum()
            q[i, valid_response_length[i]:] = 0

        for t in reversed(range(gen_len)):
            delta = q[:, t]
            lastgaelam = delta + lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = torch.zeros_like(token_level_rewards) + token_level_rewards.sum(dim=-1, keepdim=True)
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns

    pass


def compute_rloo_returns(data: verl.DataProto, eos_mask: torch.Tensor, n_samples, config):
    # calculate rloo reward on different reward sources, and sum again
    with torch.no_grad():
        discount_rewards = []
        for k, v in data.batch.items():
            if k == 'rm_scores':
                gamma = config.algorithm.adv_params.reward_model_gamma

                if config.reward_model.prime_granularity == 'whole':  # compatible for orm
                    prompt_ids = data.batch['prompts']
                    prompt_length = prompt_ids.shape[-1]
                    valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(-1)
                    reward_mask = torch.zeros_like(v, dtype=torch.bool)
                    reward_mask[torch.arange(
                        0, valid_response_length.shape[0], dtype=torch.long, device=valid_response_length.device),
                                valid_response_length - 1] = True
                else:
                    reward_mask = eos_mask.bool()
                reward_weight = config.reward_model.rm_coef
            elif k == 'gt_scores':
                gamma = config.algorithm.adv_params.verifier_gamma
                prompt_ids = data.batch['prompts']
                prompt_length = prompt_ids.shape[-1]
                valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(-1)
                reward_mask = torch.zeros_like(v, dtype=torch.bool)
                reward_mask[torch.arange(
                    0, valid_response_length.shape[0], dtype=torch.long, device=valid_response_length.device),
                            valid_response_length - 1] = True
                reward_weight = config.verifier.reward_coef
            else:  # not a reward tensor
                continue
            reward_tensor = v.clone()
            # add weighting here
            reward_tensor *= reward_weight
            reward_tensor[~reward_mask] = 0
            for start_pos in range(0, reward_tensor.shape[0], n_samples):
                cur_rewards_mean = torch.cat([
                    reward_tensor[pos:pos + 1][reward_mask[pos:pos + 1]].mean(dim=0, keepdim=True)
                    for pos in range(start_pos, start_pos + n_samples)
                ],
                                             dim=0)
                cur_rewards_sum = cur_rewards_mean.sum()
                cur_reward_baseline = cur_rewards_sum / (n_samples - 1)
                reward_tensor[start_pos:start_pos + n_samples][
                    reward_mask[start_pos:start_pos + n_samples]] = \
                    reward_tensor[start_pos:start_pos + n_samples][
                        reward_mask[start_pos:start_pos + n_samples]] * (
                                n_samples / (n_samples - 1)) - cur_reward_baseline

            discount_reward = torch.zeros_like(reward_tensor)
            for step in reversed(range(reward_tensor.shape[1])):
                if step == reward_tensor.shape[1] - 1:
                    discount_reward[:, step] = reward_tensor[:, step]
                else:
                    discount_reward[:, step] = reward_tensor[:, step] + gamma * discount_reward[:, step + 1]
            discount_rewards.append(discount_reward)
        # return is the sum of discounted reward
        returns = sum(discount_rewards)
        # advantage is whitened return
        advantages = returns.clone()
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# def compute_grpo_returns(data:verl.DataProto, eos_mask:torch.Tensor,n_samples,config):
#     with torch.no_grad():
#
#         assert 'all' in data.batch
#         reward_mask = eos_mask.bool()
#         reward_tensor = data.batch['all'].clone()
#         reward_tensor[~reward_mask] = 0
#         for start_pos in range(0, reward_tensor.shape[0],n_samples):
#             group_reward_mean = reward_tensor[start_pos:start_pos + n_samples][reward_mask[start_pos:start_pos + n_samples]].mean()
#             group_reward_std = reward_tensor[start_pos:start_pos + n_samples][reward_mask[start_pos:start_pos + n_samples]].std()
#             reward_tensor[start_pos: start_pos+n_samples][reward_mask[start_pos:start_pos+n_samples]] = (reward_tensor[start_pos: start_pos+n_samples][reward_mask[start_pos:start_pos+n_samples]]-group_reward_mean)/(1e-6+group_reward_std)
#         # original grpo does not contain discoutning
#         returns = reward_tensor.flip(dims=[1]).cumsum(-1).flip(dims=[1])
#         advantages=verl_F.masked_whiten(returns, eos_mask)
#     return returns, advantages


# optimized grpo
def compute_grpo_returns(data: verl.DataProto, eos_mask: torch.Tensor, n_samples, config):
    # calculate rloo reward on different reward sources, and sum again
    with torch.no_grad():
        discount_rewards = []
        for k, v in data.batch.items():
            if k == 'rm_scores':
                gamma = config.algorithm.adv_params.reward_model_gamma

                if config.reward_model.prime_granularity == 'whole':  # compatible for orm
                    prompt_ids = data.batch['prompts']
                    prompt_length = prompt_ids.shape[-1]
                    valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(-1)
                    reward_mask = torch.zeros_like(v, dtype=torch.bool)
                    reward_mask[torch.arange(
                        0, valid_response_length.shape[0], dtype=torch.long, device=valid_response_length.device),
                                valid_response_length - 1] = True
                else:
                    reward_mask = eos_mask.bool()
                reward_weight = config.reward_model.rm_coef
            elif k == 'gt_scores':
                gamma = config.algorithm.adv_params.verifier_gamma
                prompt_ids = data.batch['prompts']
                prompt_length = prompt_ids.shape[-1]
                valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(-1)
                reward_mask = torch.zeros_like(v, dtype=torch.bool)
                reward_mask[torch.arange(
                    0, valid_response_length.shape[0], dtype=torch.long, device=valid_response_length.device),
                            valid_response_length - 1] = True
                reward_weight = config.verifier.reward_coef
            else:  # not a reward tensor
                continue
            reward_tensor = v.clone()
            # add weighting here
            reward_tensor *= reward_weight
            reward_tensor[~reward_mask] = 0
            for start_pos in range(0, reward_tensor.shape[0], n_samples):
                cur_rewards_mean = reward_tensor[start_pos:start_pos + n_samples][reward_mask[start_pos:start_pos +
                                                                                              n_samples]].mean()
                cur_rewards_std = reward_tensor[start_pos:start_pos + n_samples][reward_mask[start_pos:start_pos +
                                                                                             n_samples]].std()

                reward_tensor[start_pos:start_pos + n_samples][
                    reward_mask[start_pos:start_pos + n_samples]] = \
                    (reward_tensor[start_pos:start_pos + n_samples][
                        reward_mask[start_pos:start_pos + n_samples]] - cur_rewards_mean) / (cur_rewards_std+1e-6)

            discount_reward = torch.zeros_like(reward_tensor)
            for step in reversed(range(reward_tensor.shape[1])):
                if step == reward_tensor.shape[1] - 1:
                    discount_reward[:, step] = reward_tensor[:, step]
                else:
                    discount_reward[:, step] = reward_tensor[:, step] + gamma * discount_reward[:, step + 1]
            discount_rewards.append(discount_reward)
        # return is the sum of discounted reward
        returns = sum(discount_rewards)
        # advantage is whitened return
        advantages = returns.clone()
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


def compute_remax_returns(data: verl.DataProto, eos_mask: torch.Tensor, n_samples, config):
    # if using remax, the first sample will always be the greedy sample. This logic will be added in the trainer class.
    with torch.no_grad():
        discount_rewards = []
        for k, v in data.batch.items():
            if k == 'rm_scores':
                gamma = config.algorithm.adv_params.reward_model_gamma
                reward_mask = eos_mask.bool()
                reward_weight = config.reward_model.rm_coef
            elif k == 'gt_scores':
                gamma = config.algorithm.adv_params.verifier_gamma
                prompt_ids = data.batch['prompts']
                prompt_length = prompt_ids.shape[-1]
                valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(-1)
                reward_mask = torch.zeros_like(v, dtype=torch.bool)
                reward_mask[torch.arange(
                    0, valid_response_length.shape[0], dtype=torch.long, device=valid_response_length.device),
                            valid_response_length - 1] = True
                reward_weight = config.verifier.reward_coef
            else:  # not a reward tensor
                continue
            reward_tensor = v.clone()
            # add weighting here
            reward_tensor *= reward_weight
            reward_tensor[~reward_mask] = 0
            for start_pos in range(0, reward_tensor.shape[0], n_samples):
                cur_reward_baseline = reward_tensor[start_pos][reward_mask[start_pos]].mean()
                reward_tensor[start_pos:start_pos + n_samples][reward_mask[start_pos:start_pos +
                                                                           n_samples]] -= cur_reward_baseline

            discount_reward = torch.zeros_like(reward_tensor)
            for step in reversed(range(reward_tensor.shape[1])):
                if step == reward_tensor.shape[1] - 1:
                    discount_reward[:, step] = reward_tensor[:, step]
                else:
                    discount_reward[:, step] = reward_tensor[:, step] + gamma * discount_reward[:, step + 1]
            discount_rewards.append(discount_reward)
        # return is the sum of discounted reward
        returns = sum(discount_rewards)
        # advantage is whitened return
        advantages = returns.clone()
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


def compute_reinforce_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """Compute policy loss for REINFORCE algorithm.

    Args:
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via REINFORCE
    """
    # Simple policy gradient loss: -log_prob * advantages
    pg_losses = -advantages * log_prob

    # Average over non-padded tokens
    pg_loss = verl_F.masked_mean(pg_losses, eos_mask)

    return pg_loss, torch.tensor(0.0), torch.tensor(0.0)


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)

    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_ce_dpo_loss_rm(token_level_scores, acc, eos_mask, beta):
    cur_scores = ((token_level_scores * eos_mask).sum(dim=1) * beta).sigmoid()
    cur_dpo_loss = torch.nn.functional.binary_cross_entropy(cur_scores, acc)
    return cur_dpo_loss

def compute_td0_loss_rm(token_level_scores, acc, eos_mask, beta):
    """

    模仿TD training，缩小两个V之间的差值，最后一个V则拟合acc
    直接用lambda=0的版本

    """
    cur_q = ((token_level_scores * eos_mask) * beta)
    loss_last = ((cur_q.sum(dim=-1)-acc)**2 / 2).sum()
    loss_inter = ((cur_q**2)**2 / 2).sum()
    return loss_last+loss_inter

def compute_td1_loss_rm(token_level_scores, acc, eos_mask, beta):
    """
    td1，所有V直接拟合acc
    """
    cur_q = ((token_level_scores * eos_mask) * beta)
    cur_q_accu = cur_q.cumsum(dim=-1)*eos_mask
    loss = (((cur_q_accu - acc.unsqueeze(-1)) * eos_mask)**2 / 2).sum()
    return loss

def compute_dpo_accuracy(token_level_scores, acc, eos_mask, n_samples):
    dpo_acc = []
    for start_id in range(0, token_level_scores.shape[0], n_samples):
        cur_scores = (token_level_scores[start_id:start_id + n_samples] *
                      eos_mask[start_id:start_id + n_samples]).sum(dim=1)

        # 求acc枚举所有样本，按照acc差值加权平均，权重全为0则设为0.5
        def get_upper_triangle(tensor_x):
            diff_matrix = tensor_x.unsqueeze(1) - tensor_x.unsqueeze(0)
            upper_tri_indices = torch.triu(torch.ones_like(diff_matrix).bool(), diagonal=1)
            return diff_matrix[upper_tri_indices]

        cur_acc_diff = get_upper_triangle(acc[start_id:start_id + n_samples])  # in range [-1,1]
        cur_score_diff = get_upper_triangle(cur_scores)  # in R
        cur_score_prediction = (cur_score_diff > 0).float()  # in [0,1]
        if cur_acc_diff.abs().sum() == 0:
            cur_acc = torch.zeros_like(cur_score_prediction[0]) + 0.5
        else:
            cur_acc = (((cur_score_diff > 0) == (cur_acc_diff > 0)).float() *
                       cur_acc_diff.abs()).sum() / cur_acc_diff.abs().sum()

        dpo_acc.append(cur_acc.unsqueeze(0))

    return torch.cat(dpo_acc, dim=0).mean()
