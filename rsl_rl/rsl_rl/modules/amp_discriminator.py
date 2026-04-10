import torch
import torch.nn as nn


class AMPDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims=None):
        super().__init__()
        hidden_dims = hidden_dims or [512, 256]

        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.SiLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, amp_obs):
        return self.network(amp_obs)

    @staticmethod
    def bce_with_logits(logits, target_is_real):
        target = torch.ones_like(logits) if target_is_real else torch.zeros_like(logits)
        return nn.functional.binary_cross_entropy_with_logits(logits, target)
