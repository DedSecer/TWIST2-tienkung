import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules import AMPDiscriminator


class AMPPPO(PPO):
    def __init__(self,
                 env,
                 actor_critic,
                 amp_obs_dim,
                 amp_disc_hidden_dims=None,
                 amp_disc_learning_rate=1e-4,
                 amp_disc_loss_coef=1.0,
                 amp_disc_grad_penalty=5.0,
                 amp_reward_scale=0.1,
                 amp_reward_clip_min=-2.0,
                 amp_reward_clip_max=2.0,
                 **kwargs):
        super().__init__(env=env, actor_critic=actor_critic, **kwargs)

        self.amp_reward_scale = amp_reward_scale
        self.amp_reward_clip_min = amp_reward_clip_min
        self.amp_reward_clip_max = amp_reward_clip_max
        self.amp_disc_loss_coef = amp_disc_loss_coef
        self.amp_disc_grad_penalty = amp_disc_grad_penalty

        self.amp_discriminator = AMPDiscriminator(
            input_dim=amp_obs_dim,
            hidden_dims=amp_disc_hidden_dims or [512, 256],
        ).to(self.device)
        self.amp_disc_optimizer = optim.Adam(self.amp_discriminator.parameters(), lr=amp_disc_learning_rate)

    def compute_amp_reward(self, amp_obs):
        with torch.no_grad():
            logits = self.amp_discriminator(amp_obs)
            prob = torch.sigmoid(logits)
            reward = -torch.log(torch.clamp(1.0 - prob, min=1e-6))
            reward = torch.clamp(reward, self.amp_reward_clip_min, self.amp_reward_clip_max)
            reward = reward.squeeze(-1) * self.amp_reward_scale
        return reward

    def _gradient_penalty(self, real_amp_obs, fake_amp_obs):
        alpha = torch.rand(real_amp_obs.shape[0], 1, device=self.device)
        interp = alpha * real_amp_obs + (1 - alpha) * fake_amp_obs
        interp.requires_grad_(True)

        interp_logits = self.amp_discriminator(interp)
        grad_outputs = torch.ones_like(interp_logits)
        grads = torch.autograd.grad(
            outputs=interp_logits,
            inputs=interp,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_norm = grads.norm(2, dim=1)
        return ((grad_norm - 1.0) ** 2).mean()

    def update_discriminator(self, policy_amp_obs, demo_amp_obs):
        fake_logits = self.amp_discriminator(policy_amp_obs)
        real_logits = self.amp_discriminator(demo_amp_obs)

        loss_fake = AMPDiscriminator.bce_with_logits(fake_logits, target_is_real=False)
        loss_real = AMPDiscriminator.bce_with_logits(real_logits, target_is_real=True)

        gp = self._gradient_penalty(demo_amp_obs, policy_amp_obs)
        disc_loss = self.amp_disc_loss_coef * (loss_fake + loss_real) + self.amp_disc_grad_penalty * gp

        self.amp_disc_optimizer.zero_grad()
        disc_loss.backward()
        nn.utils.clip_grad_norm_(self.amp_discriminator.parameters(), 1.0)
        self.amp_disc_optimizer.step()

        with torch.no_grad():
            real_acc = (torch.sigmoid(real_logits) > 0.5).float().mean()
            fake_acc = (torch.sigmoid(fake_logits) < 0.5).float().mean()
            disc_acc = 0.5 * (real_acc + fake_acc)

        return disc_loss.item(), disc_acc.item()
