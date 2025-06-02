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

def compute_prime_advantage_return(data: verl.DataProto, eos_mask: torch.Tensor, n_samples, config, dpo_acc=0.5, ):
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
        Q_tensor[:,1:]=Q_tensor[:,:-1]
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
            # 先根据正确率估计一个Q_0。如果发现Q_0和最终正确与否的信号差很远（很可能这里PRIME的reward方向都错了），那就把V_last锁定到soft bound，反向推一个Q_0出来
            # 注意log ratio本身有易负难正的特性，哪怕CE loss也很难克服这一点，可想而知nogt还不rloo就有一大堆负梯度。实验中发现了acc接近0结果reestimation也接近0的现象。需要在理论允许范围内针对性调整下，
            soft_bound = 0.99 # 需要假设reward不是0-1，否则sigmoid操作无法进行。如果需要warp，一定要注意把CE loss本身也给改掉
            M = - torch.log(1/torch.tensor(soft_bound, device=Q_tensor.device)-1)
            prepend_q0 = torch.zeros_like(data.batch['acc'])
            beta = config.reward_model.model.beta_test

            if config.algorithm.q0_estimator == 'soft':
                for i in range(0, Q_tensor.shape[0]):
                    # if (torch.sigmoid(V_last[i]+prepend_q0[i])-0.5)*(data.batch['acc'][i]-0.5)<=0: # 这个Q0估计偏了
                    if True:
                        V_target = M * (2 * data.batch['acc'][i] - 1)
                        prepend_q0[i] = V_target - V_last[i]
                        data.batch['acc'][i] = torch.sigmoid(
                            V_target)  # 理论上就该是这样，不要再保留一个小的负advantage，policy model已经可以高概率采样为正的就不用强化了
            elif config.algorithm.q0_estimator == 'mcts':
                # 根据rollout结果猜测一个q0，大体表示一个概率估计。由于bellman equation的存在。由于正常来讲只要回答里面有正的，value都会非常凸，所以这其实和设置一个大整数为baseline差别很小很小...
                if beta>0:
                    prepend_q0[:]=M
                else:
                    prepend_q0[:]=-M

            elif config.algorithm.q0_estimator == 'none':
                pass

            Q_tensor+=prepend_q0.unsqueeze(1)
            Q_tensor=torch.sigmoid(Q_tensor)

            # print('Q0 re-estimation rate: ', Q0_reestimation_count/Q_tensor.shape[0])

        # highlight: 7B 模型训练时基本上都出现了前期加快后期减慢的情况。为了尽量利用value model优势，主要是assignment方面的优势，这里需要给bias estimation加一重保险。最简单的办法就是，强行把起点和终点拉到0.5-1，如果差值不能做到这一点，就给每个token加一个小量偏差。这个操作实际上就是reward shaping，根据value function把reward分散一下，增加稳定性。

        # Q_tensor_increment = eos_mask.cumsum(dim=-1)
        # Q_tensor_increment[eos_mask==0]=0
        # Q_tensor_increment_max = Q_tensor_increment.max(dim=-1)[0]
        # Q_tensor_bias = data.batch['acc'] - torch.sigmoid(V_last)
        # Q_tensor += Q_tensor_bias.unsqueeze(-1) / Q_tensor_increment_max.unsqueeze(-1) * Q_tensor_increment


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
        # advantages = returns

        metrics=compute_value_model_metrics(Q_tensor, eos_mask, data.batch['acc'], returns)

    return advantages, returns, metrics

def compute_prime_value_advantage_return(data: verl.DataProto, eos_mask: torch.Tensor, n_samples, config):
    prompt_ids = data.batch['prompts']
    prompt_length = prompt_ids.shape[-1]
    valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(-1)
    gamma=1
    lam=config.algorithm.lam


    with torch.no_grad():
        assert 'rm_scores' in data.batch.keys() and 'acc' in data.batch.keys()
        q_tensor = data.batch['rm_scores'].clone()
        q_tensor[eos_mask==0]=0

        # 对q_tensor做rloo
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
        q_tensor = masked_rloo(q_tensor, eos_mask.bool())

        V_last = q_tensor.sum(dim=-1)
        Q_tensor = q_tensor.cumsum(dim=-1)
        Q_tensor[:,1:]=Q_tensor[:,:-1]
        Q_tensor[:,0]=0

        # normalize like prime，说实话这个norm怪怪的，不是按照value在norm而是按照dpo reward-value
        # Q_tensor/= (V_last.abs().max()+ 1e-6)
        # reverse_cumsum = torch.cumsum(q_tensor.flip(dims=[1]), dim=-1)
        # Q_tensor/= (reverse_cumsum.abs().max() + 1e-6)

        # set Q0 like prime
        Q_tensor+=(data.batch['acc']-V_last).unsqueeze(-1)
        Q_tensor[ eos_mask == 0 ] = 0

        # calculate advantage of prime with lambda
        token_level_rewards=torch.zeros_like(q_tensor)
        token_level_rewards[
            torch.arange(0, valid_response_length.shape[0], dtype=torch.long, device=valid_response_length.device),
            valid_response_length - 1] = data.batch['acc']

        lastgaelam = 0
        advantages_reversed = []
        gen_len = q_tensor.shape[1]

        for t in reversed(range(gen_len)):
            nextvalues = Q_tensor[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - Q_tensor[:, t] #
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + Q_tensor

        # calculate advantage from acc reward
        advantages_acc = eos_mask*data.batch['acc'].unsqueeze(-1)

        # combine advantages
        advantages = verl_F.masked_whiten(advantages*config.algorithm.reward_dpo_coef + advantages_acc*config.algorithm.reward_gt_coef, eos_mask)

        metrics=compute_value_model_metrics(Q_tensor, eos_mask, data.batch['acc'], returns)

    return advantages, returns, metrics

def compute_reasonable_prime_value_advantage_return(data: verl.DataProto, eos_mask: torch.Tensor, n_samples, config):
    # 假定loss是DPO loss(CE loss也没法扭转DPO loss的固有问题），
    prompt_ids = data.batch['prompts']
    prompt_length = prompt_ids.shape[-1]
    valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(-1)
    gamma=1
    lam=config.algorithm.lam


    with torch.no_grad():
        assert 'rm_scores' in data.batch.keys() and 'acc' in data.batch.keys()
        q_tensor = data.batch['rm_scores']
        q_tensor[eos_mask==0]=0
        # 直接把prm输出squeeze到0-1范围内，作用很简单就是移除DPO的negative effect，
        for i in range(0, q_tensor.shape[0], n_samples):
            q_tensor_mean = q_tensor[i:i+n_samples, :][eos_mask[i:i+n_samples].bool()].mean()
            q_tensor[i:i+n_samples] -= q_tensor_mean
        q_tensor[eos_mask==0]=0
        V_last = q_tensor.sum(dim=-1)
        Q_tensor = q_tensor.cumsum(dim=-1)
        Q_tensor[:,1:] = Q_tensor[:,:-1]
        Q_tensor[:,0]=0

        for i in range(0, Q_tensor.shape[0],n_samples):
            # estimate winrate for each prompt
            # win_rate_last = torch.sigmoid(V_last.unsqueeze(-1)-V_last.unsqueeze(0)).mean(dim=-1)
            win_rate_all = torch.sigmoid(Q_tensor[i:i+n_samples].unsqueeze(-1) - V_last[i:i+n_samples].unsqueeze(0).unsqueeze(0)).mean(dim=-1)
            global_acc = data.batch['acc'][i:i+n_samples].mean()
            # estimate_acc = 2*win_rate_all + global_acc - 1
            # estimate_acc是可能超出0-1的，不过可以优先试试不clip会发生什么
            # 另一种可行的定义方式：
            estimate_acc = (global_acc*win_rate_all)/(global_acc*win_rate_all+(1-global_acc)*(1-win_rate_all))
            Q_tensor[i:i+n_samples] = estimate_acc

        Q_tensor[ eos_mask==0 ] = 0

        # calculate advantage of prime with lambda
        token_level_rewards=torch.zeros_like(q_tensor)
        token_level_rewards[
            torch.arange(0, valid_response_length.shape[0], dtype=torch.long, device=valid_response_length.device),
            valid_response_length - 1] = data.batch['acc']

        lastgaelam = 0
        advantages_reversed = []
        gen_len = q_tensor.shape[1]

        for t in reversed(range(gen_len)):
            nextvalues = Q_tensor[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - Q_tensor[:, t] #
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + Q_tensor

        # calculate advantage from acc reward
        advantages_acc = eos_mask*data.batch['acc'].unsqueeze(-1)

        # combine advantages
        advantages = verl_F.masked_whiten(advantages*config.algorithm.reward_dpo_coef + advantages_acc*config.algorithm.reward_gt_coef, eos_mask)

        metrics=compute_value_model_metrics(Q_tensor, eos_mask, data.batch['acc'], returns)

    return advantages, returns, metrics

def compute_middle_prime_advantage_return(data: verl.DataProto, eos_mask: torch.Tensor, n_samples, config, linear=False, trust_acc=True):
    # 正例的optimal是0。比如按照acc求一个acc
    prompt_ids = data.batch['prompts']
    prompt_length = prompt_ids.shape[-1]
    valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(-1)
    gamma=1
    lam=config.algorithm.lam
    beta = config.reward_model.model.get('beta_test', 0.05)

    with torch.no_grad():
        assert 'rm_scores' in data.batch.keys() and 'acc' in data.batch.keys()
        q_tensor = data.batch['rm_scores']
        q_tensor[eos_mask==0]=0

        V_last = q_tensor.sum(dim=-1)
        Q_tensor = q_tensor.cumsum(dim=-1)
        Q_tensor[:,1:] = Q_tensor[:,:-1]
        Q_tensor[:,0]=0

        for i in range(0, Q_tensor.shape[0],n_samples):
            # estimate margin
            group_acc = data.batch['acc'][i:i + n_samples].mean()
            warped_group_acc = group_acc / 2
            margin= beta * torch.log(warped_group_acc / (1 - warped_group_acc))
            if warped_group_acc == 0:
                margin = torch.zeros_like(margin)

            Q_tensor[i:i+n_samples] = margin

        Q_tensor_reward = Q_tensor.clone()
        Q_tensor_reward[eos_mask==0]=0
        Q_tensor_reward=Q_tensor_reward.clamp(max=0.)

        # 到这一步，Q_tensor_reward是bound到(-inf, 0)的 logits数值
        # linear的含义是转化为probability
        # non linear的含义是直接sigmoid然后乘2
        if linear:
            Q_tensor = torch.exp(Q_tensor_reward/beta)
            Q_tensor[eos_mask==0]=0
        else:
            Q_tensor = torch.sigmoid(Q_tensor_reward)*2
            Q_tensor[eos_mask==0]=0

        # calculate advantage of prime with lambda
        token_level_rewards=torch.zeros_like(q_tensor)
        token_level_rewards[
            torch.arange(0, valid_response_length.shape[0], dtype=torch.long, device=valid_response_length.device),
            valid_response_length - 1] = data.batch['acc']

        lastgaelam = 0
        advantages_reversed = []
        gen_len = q_tensor.shape[1]

        for t in reversed(range(gen_len)):
            nextvalues = Q_tensor[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - Q_tensor[:, t] #
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + Q_tensor

        # calculate advantage from acc reward
        advantages_acc = eos_mask*data.batch['acc'].unsqueeze(-1)

        # combine advantages
        advantages = verl_F.masked_whiten(advantages*config.algorithm.reward_dpo_coef + advantages_acc*config.algorithm.reward_gt_coef, eos_mask)

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

            outcome_reward= data.batch['acc'].clone()
            coef = config.algorithm.reward_gt_coef
            if config.algorithm.reward_gt_coef<0: # this means that gt reward is only used to fix numerical errors. it will keep the final reward unchanged.
                reward_tensor_prime = data.batch['rm_scores']
                outcome_reward -= reward_tensor_prime.sum(dim=-1)
                coef = config.algorithm.reward_dpo_coef

            reward_mask[
                torch.arange(0, valid_response_length.shape[0], dtype=torch.long, device=valid_response_length.device),
                valid_response_length - 1] = True
            reward_tensor[
                torch.arange(0, valid_response_length.shape[0], dtype=torch.long, device=valid_response_length.device),
                valid_response_length - 1] = outcome_reward



            reward_tensors.append(masked_rloo(reward_tensor, reward_mask) * coef)

        final_reward_tensor = sum(reward_tensors)

        returns = (final_reward_tensor * eos_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        # returns = reward_tensors[1].flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1]) + reward_tensors[0]

        advantages = returns.clone()
        advantages = verl_F.masked_whiten(advantages, eos_mask)

        return advantages, returns


def compute_ce_dpo_loss_rm(token_level_scores, acc, eos_mask, beta):
    cur_scores = ((token_level_scores * eos_mask).sum(dim=1) * beta).sigmoid()
    cur_dpo_loss = torch.nn.functional.binary_cross_entropy(cur_scores, acc)
    return cur_dpo_loss

def compute_middle_ce_loss_rm(token_level_scores, acc, eos_mask, beta, margin=None):
    if margin==None:
        margin = torch.zeros_like(acc)
    cur_scores = ((token_level_scores * eos_mask).sum(dim=1) * beta + margin).sigmoid()
    cur_dpo_loss = torch.nn.functional.binary_cross_entropy(cur_scores, acc/2)
    return cur_dpo_loss

def compute_margin_ce_dpo_loss_rm(token_level_scores, acc, eos_mask, beta, gamma=1.0):
    cur_scores = ((token_level_scores * eos_mask).sum(dim=1) * beta - gamma*(acc*2-1)).sigmoid()
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
