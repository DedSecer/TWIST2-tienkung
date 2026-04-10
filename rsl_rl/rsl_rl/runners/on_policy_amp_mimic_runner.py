import os
import time
from collections import deque
from copy import copy
import statistics

import torch
import wandb

from rsl_rl.runners.on_policy_runner_mimic import OnPolicyRunnerMimic


class OnPolicyAMPMimicRunner(OnPolicyRunnerMimic):
    def __init__(self, env, train_cfg, log_dir=None, device='cpu', **kwargs):
        super().__init__(env=env, train_cfg=train_cfg, log_dir=log_dir, device=device, **kwargs)

    def learn_RL(self, num_learning_iterations, init_at_random_ep_len=False):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_disc_loss = 0.0
        mean_disc_acc = 0.0
        mean_amp_reward = 0.0
        mean_hist_latent_loss = 0.0
        mean_priv_reg_loss = 0.0
        priv_reg_coef = 0.0
        entropy_coef = 0.0
        grad_penalty_coef = 0.0

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        if self.normalize_obs:
            obs = self.normalizer.normalize(obs)
            critic_obs = self.normalizer.normalize(critic_obs) if self.critic_normalizer is None else self.critic_normalizer.normalize(critic_obs)
        infos = {}
        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        rew_explr_buffer = deque(maxlen=100)
        rew_entropy_buffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_explr_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_entropy_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        task_rew_buf = deque(maxlen=100)
        cur_task_rew_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            hist_encoding = it % self.dagger_update_freq == 0
            rollout_policy_amp_obs = []
            rollout_demo_amp_obs = []
            rollout_amp_rewards = []

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, infos, hist_encoding)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    amp_obs = infos.get("amp_obs", None)
                    amp_demo_obs = infos.get("amp_demo_obs", None)
                    if amp_obs is not None and amp_demo_obs is not None:
                        amp_obs = amp_obs.to(self.device)
                        amp_demo_obs = amp_demo_obs.to(self.device)
                        amp_reward = self.alg.compute_amp_reward(amp_obs)
                        rewards = rewards + amp_reward
                        rollout_policy_amp_obs.append(amp_obs.detach())
                        rollout_demo_amp_obs.append(amp_demo_obs.detach())
                        rollout_amp_rewards.append(amp_reward.detach())

                    if self.normalize_obs:
                        before_norm_obs = obs.clone()
                        before_norm_critic_obs = critic_obs.clone()
                        obs = self.normalizer.normalize(obs)
                        critic_obs = self.normalizer.normalize(critic_obs) if self.critic_normalizer is None else self.critic_normalizer.normalize(critic_obs)
                        if self._need_normalizer_update(it, self.alg_cfg["normalizer_update_iterations"]):
                            self.normalizer.record(before_norm_obs)
                            if self.critic_normalizer is not None:
                                self.critic_normalizer.record(before_norm_critic_obs)

                    total_rew = self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += total_rew
                        cur_reward_explr_sum += 0
                        cur_reward_entropy_sum += 0
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rew_explr_buffer.extend(cur_reward_explr_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rew_entropy_buffer.extend(cur_reward_entropy_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())

                        cur_reward_sum[new_ids] = 0
                        cur_reward_explr_sum[new_ids] = 0
                        cur_reward_entropy_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                if self.normalize_obs:
                    if self._need_normalizer_update(it, self.alg_cfg["normalizer_update_iterations"]):
                        self.normalizer.update()
                        if self.critic_normalizer is not None:
                            self.critic_normalizer.update()

                start = stop
                self.alg.compute_returns(critic_obs)

            regularization_scale = self.env.cfg.rewards.regularization_scale if hasattr(self.env.cfg.rewards, "regularization_scale") else 1
            average_episode_length = torch.mean(self.env.episode_length.float()).item() if hasattr(self.env, "episode_length") else 0
            mean_motion_difficulty = self.env.mean_motion_difficulty if hasattr(self.env, "mean_motion_difficulty") else 0

            mean_value_loss, mean_surrogate_loss, mean_priv_reg_loss, priv_reg_coef, mean_grad_penalty_loss, grad_penalty_coef = self.alg.update()

            if rollout_policy_amp_obs and rollout_demo_amp_obs:
                policy_amp_obs = torch.cat(rollout_policy_amp_obs, dim=0)
                demo_amp_obs = torch.cat(rollout_demo_amp_obs, dim=0)
                mean_amp_reward = torch.cat(rollout_amp_rewards, dim=0).mean().item()

                disc_batch_size = min(self.alg_cfg.get("amp_disc_batch_size", 8192), policy_amp_obs.shape[0], demo_amp_obs.shape[0])
                if disc_batch_size > 0:
                    policy_idx = torch.randint(0, policy_amp_obs.shape[0], (disc_batch_size,), device=self.device)
                    demo_idx = torch.randint(0, demo_amp_obs.shape[0], (disc_batch_size,), device=self.device)
                    mean_disc_loss, mean_disc_acc = self.alg.update_discriminator(policy_amp_obs[policy_idx], demo_amp_obs[demo_idx])

            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it <= 2500:
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            elif it <= 10000:
                if it % (2 * self.save_interval) == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            else:
                if it % (5 * self.save_interval) == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        super().log(locs, width=width, pad=pad)
        extra = {
            'Loss/discriminator': locs.get('mean_disc_loss', 0.0),
            'Loss/discriminator_accuracy': locs.get('mean_disc_acc', 0.0),
            'Train/mean_amp_reward': locs.get('mean_amp_reward', 0.0),
        }
        wandb.log(extra, step=locs['it'])

    def save(self, path, infos=None):
        if self.normalize_obs:
            state_dict = {
                'model_state_dict': self.alg.actor_critic.state_dict(),
                'optimizer_state_dict': self.alg.optimizer.state_dict(),
                'amp_discriminator_state_dict': self.alg.amp_discriminator.state_dict(),
                'amp_disc_optimizer_state_dict': self.alg.amp_disc_optimizer.state_dict(),
                'iter': self.current_learning_iteration,
                'normalizer': self.normalizer,
                'critic_normalizer': self.critic_normalizer,
                'infos': infos,
            }
        else:
            state_dict = {
                'model_state_dict': self.alg.actor_critic.state_dict(),
                'optimizer_state_dict': self.alg.optimizer.state_dict(),
                'amp_discriminator_state_dict': self.alg.amp_discriminator.state_dict(),
                'amp_disc_optimizer_state_dict': self.alg.amp_disc_optimizer.state_dict(),
                'iter': self.current_learning_iteration,
                'infos': infos,
            }

        torch.save(state_dict, path)
        if getattr(self.cfg, 'save_to_wandb', True):
            wandb.save(path, base_path=os.path.dirname(path))

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])

        if 'amp_discriminator_state_dict' in loaded_dict:
            self.alg.amp_discriminator.load_state_dict(loaded_dict['amp_discriminator_state_dict'])

        if self.normalize_obs and 'normalizer' in loaded_dict:
            self.normalizer = loaded_dict['normalizer']
            self.critic_normalizer = loaded_dict.get('critic_normalizer', None)

        if load_optimizer and 'optimizer_state_dict' in loaded_dict:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        if load_optimizer and 'amp_disc_optimizer_state_dict' in loaded_dict:
            self.alg.amp_disc_optimizer.load_state_dict(loaded_dict['amp_disc_optimizer_state_dict'])

        self.current_learning_iteration = int(os.path.basename(path).split('_')[1].split('.')[0])
        self.env.global_counter = self.current_learning_iteration * 24
        self.env.total_env_steps_counter = self.current_learning_iteration * 24
        return loaded_dict.get('infos', None)
