# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import statistics
from collections import defaultdict, Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pprint import pprint
from typing import Type, Dict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.dataset.rl_dataset import BufferedDataLoader

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, gamma, lam, adv_estimator, config):

    responses = data.batch['responses']
    response_length = responses.size(1)
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]
    token_level_rewards = data.batch['token_level_rewards'] if 'token_level_rewards' in list(
        data.batch.keys()) else data.batch['token_level_scores']

    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
    elif adv_estimator == 'rloo':
        # prompt_ids = data.batch['prompts']
        # prompt_length = prompt_ids.shape[-1]
        # valid_response_length = data.batch['attention_mask'][:,prompt_length:].sum(-1)
        advantages, returns = core_algos.compute_rloo_returns(data=data,
                                                              eos_mask=response_mask,
                                                              n_samples=config.data.n_samples,
                                                              config=config)
    elif adv_estimator == 'grpo':
        advantages, returns = core_algos.compute_grpo_returns(data=data,
                                                              eos_mask=response_mask,
                                                              n_samples=config.data.n_samples,
                                                              config=config)
    elif adv_estimator == 'reinforce':
        advantages, returns = core_algos.compute_reinforce_returns(
            token_level_rewards=token_level_rewards,
            eos_mask=response_mask,
            gamma=gamma,
        )
    elif adv_estimator == 'remax':
        advantages, returns = core_algos.compute_remax_returns(data=data,
                                                               eos_mask=response_mask,
                                                               n_samples=config.data.n_samples,
                                                               config=config)
    elif adv_estimator == 'gae_value':
        advantages, returns = core_algos.compute_gae_value_returns(data=data,
                                                                   eos_mask=response_mask,
                                                                   n_samples=config.data.n_samples,
                                                                   config=config,
                                                                   lam=lam,
                                                                   gamma=gamma)
    else:
        raise NotImplementedError
    data.batch['advantages'] = advantages
    data.batch['returns'] = returns
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    response_length = batch.batch['responses'].shape[-1]

    advantages = batch.batch['advantages']
    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    returns = batch.batch['returns']
    # values = batch.batch['values']

    metrics = {
        # score
        'critic/score/mean': torch.mean(sequence_score).detach().item(),
        'critic/score/max': torch.max(sequence_score).detach().item(),
        'critic/score/min': torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean': torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max': torch.max(sequence_reward).detach().item(),
        'critic/rewards/min': torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean': masked_mean(advantages, response_mask).detach().item(),
        'critic/advantages/max': torch.max(advantages[response_mask.bool()]).detach().item(),
        'critic/advantages/min': torch.min(advantages[response_mask.bool()]).detach().item(),
        # returns
        'critic/returns/mean': masked_mean(returns, response_mask).detach().item(),
        'critic/returns/max': torch.max(returns[response_mask.bool()]).detach().item(),
        'critic/returns/min': torch.min(returns[response_mask.bool()]).detach().item(),
        # values
        # 'critic/values/mean': masked_mean(values, response_mask).detach().item(),
        # 'critic/values/max': torch.max(values[response_mask.bool()]).detach().item(),
        # 'critic/values/min': torch.min(values[response_mask.bool()]).detach().item(),
        # response length
        'response_length/mean': torch.mean(response_length).detach().item(),
        'response_length/max': torch.max(response_length).detach().item(),
        'response_length/min': torch.min(response_length).detach().item(),
        # prompt length
        'prompt_length/mean': torch.mean(prompt_length).detach().item(),
        'prompt_length/max': torch.max(prompt_length).detach().item(),
        'prompt_length/min': torch.min(prompt_length).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPRIMETrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        self.train_dataloader = BufferedDataLoader(
            DataLoader(dataset=self.train_dataset,
                       batch_size=int(self.config.data.train_batch_size * self.config.data.oversample_factor),
                       shuffle=True,
                       drop_last=True,
                       collate_fn=collate_fn))

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        metric_dict = {}
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # test_batch = test_batch.to('cuda')

            # we only do validation on rule-based rm
            if test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            verifier_score, reward_metrics = self.val_reward_fn.verify(test_batch)
            reward_tensor = torch.tensor(verifier_score, dtype=torch.float32).unsqueeze(-1)

            for k, v in reward_metrics.items():
                metric_dict['test_reward/' + k] = v

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'test_score/{data_source}'] = np.mean(rewards)

        metric_dict[f'test_score/all'] = reward_tensor.mean().item()

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in ['rloo', 'remax', 'grpo', 'reinforce', 'gae_value']:
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0
        dp_size = self.actor_rollout_wg.world_size // self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        batch_size = self.config.data.train_batch_size
        n_samples = self.config.data.n_samples

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        for epoch in range(self.config.trainer.total_epochs):
            self.train_dataloader.start_new_epoch()
            while True:
                valid_batch = []
                buffer_batch = []

                if self.train_dataloader.buffer_size() > 0:
                    buffer_batch = self.train_dataloader.get_from_buffer(batch_size, self.actor_rollout_wg.world_size)
                metrics = defaultdict(list)
                metrics['timing/gen'] = 0
                metrics['timing/verify'] = 0

                while len(valid_batch) < batch_size * n_samples:
                    try:
                        batch_dict = self.train_dataloader.get_next_batch()
                    except StopIteration:
                        break

                    # generate a batch
                    with Timer(name='gen', text="{name}: {seconds:.1f} seconds") as timer:

                        newbatch: DataProto = DataProto.from_single_dict(batch_dict)

                        if len(buffer_batch) > 0:
                            newbatch = DataProto.concat([buffer_batch, newbatch])
                            print(len(newbatch))
                            buffer_batch = []

                        # the results from the same prompt should be contiguous !
                        gen_batch = newbatch.select(batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                                                    non_tensor_batch_keys={},
                                                    meta_info_keys={})

                        batch_lst = sum([[newbatch[i:i + 1] for _ in range(n_samples)] for i in range(len(newbatch))],
                                        [])

                        gen_batch.meta_info = {
                            'eos_token_id': self.tokenizer.eos_token_id,
                            'n_samples': n_samples,
                        }

                        # some algorithms require a greedy sampling
                        if self.config.algorithm.adv_estimator in ['remax']:
                            gen_batch.meta_info['n_samples'] -= 1
                            sampling_gen_batch_output = self.actor_rollout_wg.generate_sequences(prompts=gen_batch)
                            gen_batch.meta_info['n_samples'] = 1
                            gen_batch.meta_info['do_sample'] = False
                            greedy_gen_batch_output = self.actor_rollout_wg.generate_sequences(prompts=gen_batch)
                            gen_batch_list = sum([[
                                DataProto.concat([
                                    greedy_gen_batch_output[i:i + 1],
                                    sampling_gen_batch_output[i * (n_samples - 1):(i + 1) * (n_samples - 1)]
                                ])
                            ] for i in range(0, len(greedy_gen_batch_output))], [])
                            gen_batch_output = DataProto.concat(gen_batch_list)

                            # do not change the original meta_info
                            gen_batch.meta_info = {
                                'eos_token_id': self.tokenizer.eos_token_id,
                                'n_samples': n_samples,
                                'do_sample': True
                            }
                        else:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(prompts=gen_batch)

                        roll_batch = DataProto.concat(batch_lst)
                        roll_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                        roll_batch = roll_batch.union(gen_batch_output)

                    metrics['timing/gen'] += timer.last

                    # do accuracy filtering and score logging
                    with Timer(name='verify', text="{name}: {seconds:.1f} seconds") as timer:
                        scores_tensor, reward_metrics = self.reward_fn.verify(roll_batch)
                        for k, v in reward_metrics.items():
                            metrics['train_verify_score/' + k].append(v)

                        if self.config.data.filter_accuracy:
                            roll_batch = self.filter(roll_batch.batch['acc'].unsqueeze(1), roll_batch, n_samples)
                    metrics['timing/verify'] += timer.last

                    if len(valid_batch) == 0:
                        valid_batch = roll_batch
                    else:
                        valid_batch = DataProto.concat([valid_batch, roll_batch])
                    print(
                        f"collected {len(valid_batch)} / {batch_size * n_samples} rollouts and each prompt has {n_samples} responses"
                    )

                if len(valid_batch) < batch_size * n_samples:
                    break
                elif len(valid_batch) > batch_size * n_samples:
                    valid_batch = self.add_to_buffer(valid_batch, batch_size, n_samples)

                for k, v in reward_metrics.items():
                    metrics['train_verify_score/' + k] = np.mean(metrics['train_verify_score/' + k])

                batch = valid_batch
                print(f'rollout batch size: {len(batch)}')

                # careful about this! balancing batch may not be neccasery
                # self._balance_batch(batch, metrics=metrics) # highlight: this makes rloo not working!!! but ppo will still
                if self.use_reference_policy:
                    # compute reference log_prob
                    with Timer(name='ref', text="{name}: {seconds:.1f} seconds") as timer:
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)
                    metrics['timing/ref'] = timer.last

                with Timer(name='reward', text="{name}: {seconds:.1f} seconds") as timer:
                    if self.use_rm:
                        batch.meta_info['n_samples'] = n_samples
                        reward_model_tensor = self.rm_wg.compute_rm_score(batch)
                        if 'metrics' in reward_model_tensor.meta_info:
                            reward_model_metrics = reduce_metrics(reward_model_tensor.meta_info.pop('metrics'))
                            metrics.update(reward_model_metrics)
                        batch = batch.union(reward_model_tensor)

                metrics['timing/reward_model'] = timer.last

                # compute values
                if self.use_critic:
                    with Timer(name='values', text="{name}: {seconds:.1f} seconds") as timer:
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)
                    metrics['timing/values'] = timer.last

                with Timer(name='adv', text="{name}: {seconds:.1f} seconds") as timer:
                    reward_tensor_dict, reward_metrics = self.reward_fn(batch)
                    batch.batch['token_level_scores'] = reward_tensor_dict['all']
                    for k, v in reward_metrics.items():
                        metrics['train_reward/' + k] = v
                    # decomposed rewards:
                    for k, v in reward_tensor_dict.items():
                        batch.batch[k] = v

                    # compute rewards. apply_kl_penalty if available
                    batch, kl_metrics = apply_kl_penalty(batch,
                                                         kl_ctrl=self.kl_ctrl,
                                                         kl_penalty=self.config.algorithm.kl_penalty)
                    metrics.update(kl_metrics)

                    # compute advantages, executed on the driver process
                    batch = compute_advantage(batch,
                                              self.config.algorithm.gamma,
                                              self.config.algorithm.lam,
                                              adv_estimator=self.config.algorithm.adv_estimator,
                                              config=self.config)
                metrics['timing/adv'] = timer.last

                # update critic
                if self.use_critic:
                    with Timer(name='update_critic', text="{name}: {seconds:.1f} seconds") as timer:
                        critic_output = self.critic_wg.update_critic(batch)
                    metrics['timing/update_critic'] = timer.last
                    critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                    metrics.update(critic_output_metrics)

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:

                    # if there is an adaptive entropy controller, learn the entropy_coef here. because the batch size is quite large, we set the learning rate to 0.01
                    if self.config.actor_rollout_ref.actor.entropy_mode == 'adaptive':

                        if self.global_steps == 0:
                            entropy_coeff = 0
                        batch.meta_info['entropy_coeff'] = entropy_coeff
                        metrics['actor/entropy_coeff'] = entropy_coeff

                    # update actor
                    with Timer(name='update_actor', text="{name}: {seconds:.1f} seconds") as timer:
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    metrics['timing/update_actor'] = timer.last
                    actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                    metrics.update(actor_output_metrics)

                    # update entropy coeff
                    if self.config.actor_rollout_ref.actor.entropy_mode == 'adaptive':
                        entropy_target = self.config.actor_rollout_ref.actor.entropy_coeff
                        # response_length = batch.batch['responses'].shape[-1]
                        # log_prob = batch.batch['old_log_probs']
                        # eos_mask = batch.batch['attention_mask'][:, -response_length:]
                        # entropy_current = core_algos.compute_entropy_loss(log_prob, eos_mask)
                        entropy_current = metrics['actor/entropy_loss']
                        entropy_coeff += 0.01 * (entropy_target - entropy_current
                                                )  # if entropy is too small, we should increase this coefficient

                # validate
                if self.val_reward_fn is not None and (self.global_steps + 1) % self.config.trainer.test_freq == 0:
                    with Timer(name='testing', text="{name}: {seconds:.1f} seconds") as timer:
                        val_metrics: dict = self._validate()
                        val_metrics = {f'val/{key}': val for key, val in val_metrics.items()}
                    metrics['timing/testing'] = timer.last
                    metrics.update(val_metrics)

                # collect metrics
                data_metrics = compute_data_metrics(batch=batch)
                metrics.update(data_metrics)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if self.config.trainer.save_freq > 0 and (self.global_steps + 1) % self.config.trainer.save_freq == 0:
                    actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                                    f'global_step_{self.global_steps}')
                    actor_remote_path = None  #if self.config.trainer.default_hdfs_dir is None else os.path.join(
                    # self.config.trainer.default_hdfs_dir, 'actor')
                    self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

                    if self.use_critic:
                        critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                                         f'global_step_{self.global_steps}')
                        critic_remote_path = None  #if self.config.trainer.default_hdfs_dir is None else os.path.join(
                        # self.config.trainer.default_hdfs_dir, 'critic')
                        self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

                    if self.use_rm and self.config.reward_model.prime_model.update != 'none':
                        rm_local_path = os.path.join(self.config.trainer.default_local_dir, 'reward',
                                                     f'global_step_{self.global_steps}')
                        rm_remote_path = None
                        self.rm_wg.save_checkpoint(rm_local_path, rm_remote_path)
                self.global_steps += 1

        # perform validation after training
        if self.val_reward_fn is not None:
            val_metrics = self._validate()
            pprint(f'Final validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)

    def filter(self, reward_tensor, batch, n_samples):
        reward_matrix = reward_tensor.sum(-1).reshape(-1, n_samples)
        # reward_matrix = reward_tensor.sum(-1).reshape(n_samples, -1).T
        acc_tensor = torch.mean(reward_matrix, dim=-1)
        counts = Counter(acc_tensor.tolist())
        print(" ".join(f"{k}:{v}" for k, v in sorted(counts.items())))
        # print(acc_tensor)
        acc_mask = (acc_tensor >= self.config.data.accuracy_lower_bound) & (acc_tensor
                                                                            <= self.config.data.accuracy_upper_bound)
        acc_mask = acc_mask.repeat_interleave(n_samples)
        batch = batch.slice(acc_mask)
        return batch

    def add_to_buffer(self, batch, batch_size, n_samples):
        buffer_length = len(batch) // n_samples - batch_size
        buffer_batch = batch.slice(range(batch_size * n_samples, (buffer_length + batch_size) * n_samples, n_samples))
        # notice that we only add prompts to buffer, and slicing strategy should be exactly consistent to what is in ray_trainer.py
        buffer_batch = buffer_batch.select(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
        buffer_batch.slice_batch(start=0, length=self.config.data.max_prompt_length, dim=1)
        buffer_mask = torch.ones(buffer_length + batch_size, dtype=torch.bool)
        buffer_mask[batch_size:] = False
        buffer_mask = buffer_mask.repeat_interleave(n_samples)
        batch = batch.slice(buffer_mask)
        self.train_dataloader.add_to_buffer(buffer_batch)
        return batch
