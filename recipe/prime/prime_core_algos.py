# Copyright 2024 PRIME team and/or its affiliates
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

import torch
import verl
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import compute_value_model_metrics

def compute_prime_advantage_return(data: verl.DataProto, eos_mask: torch.Tensor, n_samples, config, dpo_acc=0.5):
    # 把PRIME的输出当做value model来用，这会有一个partition项需要估计，这里简单处理就直接平均作差完事。
    # 然后再来GAE。可以确保这样做和PRIME等同。
    prompt_ids = data.batch['prompts']
    prompt_length = prompt_ids.shape[-1]
    valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(-1)
    gamma=1
    lam=config.algorithm.lam

    with torch.no_grad():
        assert 'rm_scores' in data.batch.keys() and 'acc' in data.batch.keys()
        q_tensor = data.batch['rm_scores']
        q_tensor[eos_mask==0]=0
        V_last = q_tensor.sum(dim=-1)
        Q_tensor = q_tensor.cumsum(dim=-1)
        Q_tensor[:,1:]=q_tensor[:,:-1]
        Q_tensor[:,0]=0
        # for start_pos in range(0,q_tensor.shape[0], n_samples):
            # highlight: partition暂时被修改，partition总是直接等于acc-value_last，加上这个以后可以让prime在训崩以后还能重新把acc拉起来
            # highlight 2: BT model和reward model之间确实无法直接统一，而且BT unbound似乎真的引入了一些问题。在这里出于稳定性的考虑，将iprm的输出首先norm到0-1范围内。多少属于没有办法的办法
            # highlight 3: 直接把温度缩放拿出来用，这是直接由BT的特性导出的。平均最后一个token的得分，来寻找计算结果应该用的参照得分是多少
            # highlight 4: 利用整体acc来求baseline，注意这是个有偏估计
            # partition = (data.batch['acc'][start_pos:start_pos+n_samples] - V_last[start_pos:start_pos+n_samples])
            # Q_tensor[start_pos:start_pos+n_samples] += partition.unsqueeze(-1)
            # Q_tensor[start_pos:start_pos+n_samples] = (Q_tensor[start_pos:start_pos+n_samples]-Q_tensor[start_pos:start_pos+n_samples].min())/(Q_tensor[start_pos:start_pos+n_samples].max()-Q_tensor[start_pos:start_pos+n_samples].min())

            # avg_score = V_last[start_pos:start_pos+n_samples].mean()
            # avg_reward = data.batch['acc'][start_pos:start_pos+n_samples].mean() # 注意偶尔avg_reward会是0
            # avg_reward=torch.clamp(avg_reward,1/n_samples,1-1/n_samples)
            # baseline_score = avg_score + torch.log((1-avg_reward)/avg_reward)
            # Q_tensor[start_pos:start_pos+n_samples]=torch.sigmoid(Q_tensor[start_pos: start_pos+n_samples]-baseline_score)
        if config.reward_model.model.loss_type == 'dpo':
            avg_score = V_last.mean()
            avg_reward=data.batch['acc'].mean()
            baseline_score=avg_score+torch.log((1-avg_reward)/avg_reward)
            Q_tensor = torch.sigmoid(Q_tensor-baseline_score)
        else:
            Q_tensor = torch.sigmoid(Q_tensor)

        Q_tensor[eos_mask==0]=0
        # V(t) = Q(t-1)，V_0应该总是partition，相当于V_value需要把Q整体后移一位才对
        # 注意Q_tensor的含义是V，不要搞混

        # value clipping: 由于这里的Q优化比较unbound，需要强制锁定clip到0-1范围内，避免各种不稳定
        # Q_tensor = torch.clamp(Q_tensor, min=0, max=1)

        # value normalizing: 不允许出现不合理的value，所以每个回答的value需要被norm到0和1之间
        # Q_max = Q_tensor.max(dim=-1)[0].unsqueeze(-1)
        # Q_max[Q_max<1]=1
        # Q_tensor /= Q_max
        # Q_tensor[eos_mask == 0] = 0
        #
        # Q_min = Q_tensor.min(dim=-1)[0].unsqueeze(-1)
        # Q_min[Q_min>0]=0
        # Q_tensor = 1-(1-Q_tensor)/(1-Q_min)
        # Q_tensor[eos_mask == 0] = 0



        # reward tensor在这里需要被保留
        token_level_rewards=torch.zeros_like(q_tensor)
        token_level_rewards[
            torch.arange(0, valid_response_length.shape[0], dtype=torch.long, device=valid_response_length.device),
            valid_response_length - 1] = data.batch['acc']

        lastgaelam = 0
        advantages_reversed = []
        gen_len = q_tensor.shape[1]

        for t in reversed(range(gen_len)):
            nextvalues = Q_tensor[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - Q_tensor[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + Q_tensor
        advantages = verl_F.masked_whiten(advantages, eos_mask)

        metrics=compute_value_model_metrics(Q_tensor, eos_mask, data.batch['acc'], returns)

    return advantages, returns, metrics


def compute_rloo_advantage_return(data: verl.DataProto, eos_mask: torch.Tensor, n_samples, config):
    # calculate rloo reward on different reward sources, and sum again
    def masked_rloo(reward_tensor_original, mask_tensor):
        reward_tensor = reward_tensor_original.clone()
        reward_tensor[~mask_tensor] = 0
        for start_pos in range(0, reward_tensor.shape[0], n_samples):
            cur_rewards_mean = torch.cat([
                reward_tensor[pos:pos + 1][mask_tensor[pos:pos + 1]].mean(dim=0, keepdim=True)
                for pos in range(start_pos, start_pos + n_samples)
            ],
                                         dim=0)
            cur_rewards_sum = cur_rewards_mean.sum()
            cur_reward_baseline = cur_rewards_sum / (n_samples - 1)
            reward_tensor[start_pos:start_pos + n_samples][
                mask_tensor[start_pos:start_pos + n_samples]] = \
                reward_tensor[start_pos:start_pos + n_samples][
                    mask_tensor[start_pos:start_pos + n_samples]] * (
                        n_samples / (n_samples - 1)) - cur_reward_baseline

        return reward_tensor

    reward_tensors = []

    with torch.no_grad():

        if 'rm_scores' in data.batch.keys() and config.algorithm.reward_dpo_coef != 0.:
            reward_tensor = data.batch['rm_scores']
            reward_mask = eos_mask.bool()

            reward_tensors.append(masked_rloo(reward_tensor, reward_mask) * config.algorithm.reward_dpo_coef)

        if 'acc' in data.batch.keys() and config.algorithm.reward_gt_coef != 0.:
            reward_tensor = torch.zeros_like(eos_mask, dtype=torch.float32)
            reward_mask = torch.zeros_like(eos_mask, dtype=torch.bool)

            prompt_ids = data.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(-1)

            reward_mask[
                torch.arange(0, valid_response_length.shape[0], dtype=torch.long, device=valid_response_length.device),
                valid_response_length - 1] = True
            reward_tensor[
                torch.arange(0, valid_response_length.shape[0], dtype=torch.long, device=valid_response_length.device),
                valid_response_length - 1] = data.batch['acc']

            reward_tensors.append(masked_rloo(reward_tensor, reward_mask) * config.algorithm.reward_gt_coef)

        final_reward_tensor = sum(reward_tensors)

        returns = (final_reward_tensor * eos_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        advantages = returns.clone()
        advantages = verl_F.masked_whiten(advantages, eos_mask)

        return advantages, returns


def compute_ce_dpo_loss_rm(token_level_scores, acc, eos_mask, beta):
    cur_scores = ((token_level_scores * eos_mask).sum(dim=1) * beta).sigmoid()
    cur_dpo_loss = torch.nn.functional.binary_cross_entropy(cur_scores, acc)
    return cur_dpo_loss


def compute_detach_dpo_loss_rm(token_level_scores, acc, Q_bc, acc_bc, eos_mask, beta, bon_mode='none', use_ce=False):
    # we always assume that the BoN size equals n_samples
    # mode1: use acc as rm
    # mode2: use Q as rm
    cur_Q = (token_level_scores * eos_mask).sum(dim=1) * beta
    other_Q = torch.zeros_like(cur_Q)
    batch_size = token_level_scores.shape[0]
    for i in range(token_level_scores.shape[0]):
        if acc[i] > 0:
            Q_chosen = Q_bc[i][acc_bc[i] < acc[i]]
        else:
            Q_chosen = Q_bc[i][acc_bc[i] > acc[i]]
        if use_ce or len(Q_chosen) == 0:
            other_Q[i] = 0
        else:
            other_Q[i] = Q_chosen.mean() * beta

    dpo_loss = -torch.log(torch.sigmoid((cur_Q - other_Q) * ((acc > 0).float() * 2 - 1)))
    if bon_mode == 'none':
        dpo_loss = dpo_loss.mean()
    else:
        weight = torch.zeros_like(dpo_loss)
        n_samples = acc_bc.shape[1]
        n_samples_valid = 2
        if bon_mode == 'bon_rm':
            for i in range(token_level_scores.shape[0]):
                cur_response_weights = n_samples_valid * torch.pow(
                    (Q_bc[i].unsqueeze(0) <= Q_bc[i].unsqueeze(1)).float().mean(dim=-1), n_samples_valid - 1)
                weight[i] = n_samples_valid * torch.pow(
                    (Q_bc[i] * beta <= cur_Q[i]).float().mean(),
                    n_samples_valid - 1) / cur_response_weights.sum() * n_samples / batch_size
        elif bon_mode == 'bon_acc':
            for i in range(token_level_scores.shape[0]):
                cur_response_weights = n_samples_valid * torch.pow(
                    (acc_bc[i].unsqueeze(0) <= acc_bc[i].unsqueeze(1)).float().mean(dim=-1), n_samples_valid - 1)
                weight[i] = n_samples_valid * torch.pow(
                    (acc_bc[i] <= acc[i]).float().mean(),
                    n_samples_valid - 1) / cur_response_weights.sum() * n_samples / batch_size

        else:
            raise NotImplementedError
        dpo_loss = (dpo_loss * weight).sum()

    return dpo_loss


def compute_dpo_accuracy(token_level_scores, acc, eos_mask, n_samples):
    dpo_acc = []
    for start_id in range(0, token_level_scores.shape[0], n_samples):
        cur_scores = (token_level_scores[start_id:start_id + n_samples] *
                      eos_mask[start_id:start_id + n_samples]).sum(dim=1)

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

def compute_dpo_continual_accuracy(token_level_scores, acc, eos_mask, n_samples): # 确定连续正确性的高低，后面用来做platt calibration，基本假设是现有模型对于每个sample的continual acc一致
    dpo_acc = []
    for start_id in range(0, token_level_scores.shape[0], n_samples):
        cur_scores = (token_level_scores[start_id:start_id + n_samples] *
                      eos_mask[start_id:start_id + n_samples]).sum(dim=1)

        def get_upper_triangle(tensor_x):
            diff_matrix = tensor_x.unsqueeze(1) - tensor_x.unsqueeze(0)
            upper_tri_indices = torch.triu(torch.ones_like(diff_matrix).bool(), diagonal=1)
            return diff_matrix[upper_tri_indices]

        cur_acc_diff = get_upper_triangle(acc[start_id:start_id + n_samples])  # in range [-1,1]
        cur_score_diff = get_upper_triangle(cur_scores)  # in R
        cur_score_diff_signed = cur_score_diff*cur_acc_diff # 同对同错的，score会变成0，正确率是0.5
        cur_acc = torch.sigmoid(cur_score_diff_signed)

        dpo_acc.append(cur_acc.unsqueeze(0))

    return torch.cat(dpo_acc, dim=0).mean()
def compute_dpo_abs_accuracy(token_level_scores, acc, eos_mask, n_samples):
    return (torch.sign((token_level_scores * eos_mask).sum(dim=-1)) == torch.sign(acc * 2 - 1)).float().mean()


def compute_return_abs_accuracy(returns, acc):
    return (torch.sign(returns[:, 0]) == torch.sign(acc * 2 - 1)).float().mean()


def compute_return_smoothness(returns):
    return ((returns[:, :-1] - returns[:, 1:])**2).sum(dim=-1).mean()
