"""A conditional pixel DDPM and its Gaussian transition policy."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from mnist_data import SIZE


class ResidualBlock(nn.Module):
    def __init__(self, channels, context_width):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, channels)
        self.norm2 = nn.GroupNorm(4, channels)
        self.context = nn.Linear(context_width, 2 * channels)

    def forward(self, images, conditioning):
        hidden = self.conv1(F.silu(self.norm1(images)))
        scale, offset = self.context(conditioning).chunk(2, dim=1)
        hidden = self.norm2(hidden) * (1 + scale[:, :, None, None]) + offset[:, :, None, None]
        return images + self.conv2(F.silu(hidden))


class Denoiser(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        context_width = 4 * width
        channels = 2 * width
        self.register_buffer('frequencies', torch.exp(torch.linspace(0.0, -6.0, 16)))
        self.digit = nn.Embedding(11, 32)
        self.color = nn.Embedding(4, 16)
        self.side = nn.Embedding(3, 8)
        self.condition = nn.Sequential(nn.Linear(88, context_width), nn.SiLU(), nn.Linear(context_width, context_width))
        self.input = nn.Conv2d(3, width, 3, padding=1)
        self.down1 = nn.Conv2d(width, channels, 4, stride=2, padding=1)
        self.down2 = nn.Conv2d(channels, channels, 4, stride=2, padding=1)
        self.encode1 = ResidualBlock(width, context_width)
        self.encode2 = ResidualBlock(channels, context_width)
        self.middle = ResidualBlock(channels, context_width)
        self.spatial_condition = nn.Linear(context_width, channels * (SIZE // 4) ** 2)
        self.up1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.up2 = nn.Conv2d(channels, width, 3, padding=1)
        self.decode1 = ResidualBlock(channels, context_width)
        self.decode2 = ResidualBlock(width, context_width)
        self.output = nn.Sequential(nn.GroupNorm(4, width), nn.SiLU(), nn.Conv2d(width, 3, 3, padding=1))

    def forward(self, images, times, prompts):
        angles = times.float()[:, None] * self.frequencies
        features = torch.cat([
            angles.sin(), angles.cos(), self.digit(prompts[:, 0]),
            self.color(prompts[:, 1]), self.side(prompts[:, 2]),
        ], dim=1)
        context = self.condition(features)
        full = self.encode1(self.input(images), context)
        half = self.encode2(self.down1(full), context)
        quarter = self.down2(half)
        prompt_map = self.spatial_condition(context).reshape(len(images), quarter.shape[1], SIZE // 4, SIZE // 4)
        if prompt_map.shape[-1] != quarter.shape[-1]:
            prompt_map = F.interpolate(prompt_map, size=quarter.shape[-2:], mode='bilinear', align_corners=False)
        quarter = self.middle(quarter + prompt_map, context)
        hidden = self.decode1(self.up1(F.interpolate(quarter, scale_factor=2, mode='nearest')) + half, context)
        hidden = self.decode2(self.up2(F.interpolate(hidden, scale_factor=2, mode='nearest')) + full, context)
        return self.output(hidden)


class Diffusion:
    def __init__(self, steps, device, image_size=SIZE):
        self.steps = steps
        self.image_size = image_size
        grid = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
        cumulative = torch.cos(((grid / steps + 0.008) / 1.008) * math.pi / 2).square()
        cumulative = cumulative / cumulative[0]
        betas = (1 - cumulative[1:] / cumulative[:-1]).clamp(max=0.999).float().to(device)
        alphas = 1 - betas
        self.alpha_bar = alphas.cumprod(0)
        previous = torch.cat([torch.ones(1, device=device), self.alpha_bar[:-1]])
        self.signal = self.alpha_bar.sqrt()
        self.noise = (1 - self.alpha_bar).sqrt()
        self.clean_coef = betas * previous.sqrt() / (1 - self.alpha_bar)
        self.state_coef = (1 - previous) * alphas.sqrt() / (1 - self.alpha_bar)
        self.variance = betas * (1 - previous) / (1 - self.alpha_bar)

    @staticmethod
    def value(values, times):
        return values[times, None, None, None]

    def supervised_loss(self, model, clean, prompts, objective='mse'):
        times = torch.randint(self.steps, (len(clean),), device=clean.device)
        noise = torch.randn_like(clean)
        a, s = self.value(self.signal, times), self.value(self.noise, times)
        prediction = model(a * clean + s * noise, times, prompts)
        error = (prediction - clean).square()
        if objective == 'foreground':
            weights = 1 + 3 * clean.add(1).mul(0.5)
            return (error * weights).mean() / weights.mean()
        if objective == 'edge':
            horizontal = prediction.diff(dim=-1) - clean.diff(dim=-1)
            vertical = prediction.diff(dim=-2) - clean.diff(dim=-2)
            return error.mean() + 0.35 * (horizontal.square().mean() + vertical.square().mean())
        return error.mean()

    def mean(self, state, times, clean_prediction):
        clean = clean_prediction.clamp(-1, 1)
        return self.value(self.clean_coef, times) * clean + self.value(self.state_coef, times) * state

    @torch.no_grad()
    def rollout(self, model, prompts, generator=None, trajectory=False, paired_noise=False):
        noise_device = prompts.device if generator is None else generator.device
        def sample_noise():
            size = 32 if paired_noise else self.image_size
            value = torch.randn((len(prompts), 3, size, size), device=noise_device, generator=generator)
            if size != self.image_size:
                value = F.avg_pool2d(value, 2) * 2
            return value.to(prompts.device)

        state = sample_noise()
        states = []
        for step in reversed(range(self.steps)):
            times = torch.full((len(prompts),), step, device=prompts.device, dtype=torch.long)
            if trajectory:
                states.append(state)
            mean = self.mean(state, times, model(state, times, prompts))
            if step > 0:
                noise = sample_noise()
                state = mean + self.variance[step].sqrt() * noise
            else:
                state = mean
        images = state.add(1).div(2).clamp(0, 1)
        return images, torch.stack(states[::-1]) if trajectory else None

    def sample_states(self, trajectory, places, skip_deterministic=False):
        """Draw transition indices uniformly and keep their prompt/trajectory index."""
        _, batch, *_ = trajectory.shape
        indices = torch.arange(batch, device=trajectory.device).repeat(places)
        times = torch.randint(int(skip_deterministic), self.steps, indices.shape, device=trajectory.device)
        current = trajectory[times, indices]
        return current, times, indices
