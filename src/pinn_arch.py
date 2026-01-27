# Geometry-Aware Parametric PINN
# Inputs: (x, y, z, t, heat_flux, T_initial, L1, L2, H, thickness, volume, surface_area, S_V_ratio) -> Temperature
import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, hidden_dim),  # 13 inputs: spatial(3) + time(1) + process(2) + geometry(7)
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs):
        return self.net(inputs)
