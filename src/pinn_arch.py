# Parametric PINN: (x, y, z, t, heat_flux, T_initial) -> Temperature
import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden_dim),  # 6 inputs: x, y, z, t, heat_flux, T_initial
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs):
        return self.net(inputs)
