import torch 
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
import time

# Define ODE system for the reaction
class ReactionODE(nn.Module):
    def __init__(self, H2SO4, Temperature):
        """
        Initialize the ODE system with sulfuric acid concentration and temperature.

        Args:
            H2SO4 (float): Sulfuric acid concentration.
            Temperature (float): Temperature in Celsius.
        """
        super().__init__()
        self.H2SO4 = H2SO4
        self.Temperature = Temperature + 273.15  # Convert to Kelvin
        self.R = 8.314  # Universal gas constant
        self.n = 0.85   # Reaction parameter

    def forward(self, t, C):
        """
        Defines the ODE system.

        Args:
            t (Tensor): Time (not used explicitly as ODE is autonomous).
            C (Tensor): Concentrations [NB, NO2, H2SO4].

        Returns:
            Tensor: Time derivatives of concentrations.
        """
        k = torch.exp(61.48 - 126813 / (self.R * self.Temperature))
        MC = (-(2.16e-4 * self.H2SO4**5 - 1.27e-2 * self.H2SO4**4 + 
                0.28 * self.H2SO4**3 - 2.73 * self.H2SO4**2 + 10.6 * self.H2SO4)
              * (200 / self.Temperature + 0.3292))
        rate = -k * C[0] * C[1] * 10 ** (self.n * MC)
        dCdt = torch.tensor([rate, 0.0, 0.0], dtype=torch.float32)  # Only NB changes
        return dCdt


def reaction_torch(X):
    """
    Solve the ODE system and compute conversion and E-factor.

    Args:
        X (Tensor): Input variables [Time, Temp, NB, NO2, H2SO4].

    Returns:
        Tuple[Tensor, Tensor]: (E-factor, conversion).
    """
    Time, Temp, NB, NO2, H2SO4 = X
    C0 = torch.tensor([NB, NO2, H2SO4], dtype=torch.float32)
    func = ReactionODE(H2SO4, Temp)
    t = torch.linspace(0., Time, steps=20)
    C_final = odeint(func, C0, t, rtol=1e-4, atol=1e-6)[-1]

    NB0 = NB
    NBt = C_final[0]
    conversion = (NB0 - NBt) / NB0
    conversion = torch.clamp(conversion, 0, 1)  # Ensure within [0,1]

    e_factor = (Time / 200 + Temp / 82 + H2SO4 / 18.6 + torch.abs(NO2 - NB) / NB) / torch.exp(NB * conversion)

    return e_factor, conversion


def run_pareto(num_points=20, steps=100, lr=0.01):
    """
    Run multi-objective optimization with weighted sum method to approximate Pareto front.

    Args:
        num_points (int): Number of Pareto points to generate.
        steps (int): Optimization steps for each weight vector.
        lr (float): Learning rate for optimizer.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (Pareto-optimal input points, corresponding objective values).
    """
    X_bounds = torch.tensor([[0, 200], [60, 82], [0.01, 1], [0.01, 2], [15, 18.4]])  # Design space
    pareto_X, pareto_Y = [], []

    # Generate evenly spaced weight preferences
    prefs = torch.tensor([[w, 1 - w] for w in np.linspace(0, 1, num_points)], dtype=torch.float32)

    start_time = time.time()

    for i, (w1, w2) in enumerate(prefs):
        init = torch.tensor([100.0, 70.0, 0.5, 1.0, 16.0])
        scale = (X_bounds[:, 1] - X_bounds[:, 0]) * 0.1  # 🔧 控制扰动比例（可调）
        noise = torch.randn(5) * scale
        x = (init + noise).clamp(X_bounds[:, 0], X_bounds[:, 1]).detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=lr)
        

        # Optimize weighted objective
        for step in range(steps):
            optimizer.zero_grad()
            ef, yld = reaction_torch(x)
            loss = w1 * ef + w2 * (-yld)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                # Clamp values within bounds
                x.data = torch.max(torch.min(x, X_bounds[:, 1]), X_bounds[:, 0])

        with torch.no_grad():
            ef, yld = reaction_torch(x)
            pareto_X.append(x.detach().numpy())
            pareto_Y.append([ef.item(), yld.item()])

        if i % 10 == 0:
            print(f'[{i}/{num_points}] E-factor={ef:.4f}, Conversion={yld:.4f}')

    end_time = time.time()
    print(f"\n Total time: {end_time - start_time:.2f} seconds")
    return np.array(pareto_X), np.array(pareto_Y)


# Run optimization and plot Pareto front
pareto_X, pareto_Y = run_pareto(num_points=20, steps=100, lr=0.01)

plt.figure(figsize=(6, 4))
plt.scatter(pareto_Y[:, 1], pareto_Y[:, 0], c='blue', s=50)
plt.ylabel("E-factor (minimize)")
plt.xlabel("Conversion (maximize)")
plt.title("Pareto Front with Gradient-based methods")
plt.grid(True)
plt.tight_layout()
plt.savefig("pareto_mtl.png", dpi=300)  # Save at high resolution
plt.show()