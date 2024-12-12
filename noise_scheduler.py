import math 
import numpy as np 
import torch 

# noise scheduler 
class LinearNoiseEscheduler:
    def __init__(self, num_timesteps, beta_start, beta_end) -> None:
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas 
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)

    def add_noise(self, original, noise, t):
        original_shape = original.shape
        batch_size = original_shape[0]

        # sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].reshape(batch_size)
        sqrt_alpha_cum_prod = torch.full((batch_size,), self.sqrt_alpha_cum_prod[t])
        # sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = torch.full((batch_size,), self.sqrt_one_minus_alpha_cum_prod[t])

        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)
        
        return sqrt_alpha_cum_prod * original + sqrt_one_minus_alpha_cum_prod * noise
    
    def sample_prev_timestep(self, xt, noise_pred, t):
        x0 = (xt - (self.sqrt_one_minus_alpha_cum_prod[t] * noise_pred)) / self.sqrt_alpha_cum_prod[t]
        x0 = torch.clamp(x0, -1., 1.)

        mean = xt - ((self.betas[t] * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod[t]))
        mean = mean / torch.sqrt(self.alphas[t])

        if t == 0:
            return mean, x0 
        else:
            variance = (1 - self.alpha_cum_prod[t-1]) / (1 - self.alpha_cum_prod[t])
            variance = variance * self.betas[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma * z, x0

class CosineNoiseScheduler:
    def __init__(self, num_timesteps, s=0.008):
        self.num_timesteps = num_timesteps
        # Create a normalized time scale from 0 to 1
        steps = torch.arange(num_timesteps, dtype=torch.float32)
        t = steps / (num_timesteps - 1)

        # Compute alpha_bar (cumulative alpha) using cosine schedule
        # alpha_bar_t = (cos((t+s)/(1+s) * (pi/2)))^2 / (cos(s/(1+s)*(pi/2)))^2
        alpha_bar = (torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2) / \
                    (np.cos(s / (1 + s) * math.pi / 2) ** 2)

        # alpha_bar is essentially alpha_cum_prod
        self.alpha_cum_prod = alpha_bar

        # Compute individual alphas from alpha_cum_prod
        self.alphas = torch.empty_like(alpha_bar)
        self.alphas[0] = self.alpha_cum_prod[0]
        self.alphas[1:] = self.alpha_cum_prod[1:] / self.alpha_cum_prod[:-1]

        # Compute betas from alphas
        self.betas = 1 - self.alphas

        # Precompute square roots for convenience
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)

    def add_noise(self, original, noise, t):
        """
        Add noise to the input 'original' at timestep t using the cosine schedule.
        """
        original_shape = original.shape
        batch_size = original_shape[0]

        sqrt_alpha_cum_prod = torch.full((batch_size,), self.sqrt_alpha_cum_prod[t], device=original.device)
        sqrt_one_minus_alpha_cum_prod = torch.full((batch_size,), self.sqrt_one_minus_alpha_cum_prod[t], device=original.device)

        # Reshape these to match the dimensions of 'original'
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        # Noisy sample
        return sqrt_alpha_cum_prod * original + sqrt_one_minus_alpha_cum_prod * noise

    def sample_prev_timestep(self, xt, noise_pred, t):
        """
        Sample from the previous timestep given xt and predicted noise at timestep t.
        """
        # Compute x0 estimate
        x0 = (xt - (self.sqrt_one_minus_alpha_cum_prod[t] * noise_pred)) / self.sqrt_alpha_cum_prod[t]
        x0 = torch.clamp(x0, -1., 1.)

        # Compute mean of the posterior distribution q(x_{t-1} | x_t, x_0)
        mean = xt - ((self.betas[t] * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod[t]))
        mean = mean / torch.sqrt(self.alphas[t])

        if t == 0:
            # No noise added at the last step
            return mean, x0
        else:
            # Add variance
            variance = (1 - self.alpha_cum_prod[t-1]) / (1 - self.alpha_cum_prod[t]) * self.betas[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape, device=xt.device)
            return mean + sigma * z, x0
