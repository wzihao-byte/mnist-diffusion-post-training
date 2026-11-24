"""Train a common OCR evaluator on MNIST glyphs disjoint from generator training."""

import torch
from torch import nn
from torch.nn import functional as F

from ablate import inputs_at
from mnist_data import random_prompts, render
from shape import as_size, geometry
from train import frozen, read_weights, save_weights, seed_all, update


def classifier_input(images):
    _, ink, _, x_center = geometry(images)
    size = images.shape[-1]
    shift = (x_center - (size - 1) / 2).round().long()
    columns = torch.arange(size, device=images.device)[None, :] + shift[:, None]
    valid = (columns >= 0) & (columns < size)
    return ink.gather(-1, columns.clamp(0, size - 1)[:, None, :].expand_as(ink)).mul(valid[:, None, :])[:, None] / 0.85


def new_reader():
    return nn.Sequential(
        nn.Conv2d(1, 20, 3, padding=1), nn.SiLU(), nn.MaxPool2d(2),
        nn.Conv2d(20, 32, 3, padding=1), nn.SiLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 48, 3, padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(), nn.Linear(48 * 4 * 4, 96), nn.SiLU(), nn.Linear(96, 10),
    )


def train_judge(data_dir, seed, directory, device):
    model = new_reader().to(device)
    if (directory / 'common_evaluator.pt').exists():
        return frozen(read_weights(directory, 'common_evaluator', model))
    _, _, low, _ = inputs_at(data_dir, seed, 16, device)
    _, _, high, _ = inputs_at(data_dir, seed, 32, device)
    seed_all(seed + 17)
    model = new_reader().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0017)
    for step in range(700):
        glyphs, labels = (low, high)[step % 2]
        size = 16 if step % 2 == 0 else 32
        indices = torch.randint(len(glyphs), (128,), device=device)
        prompt = random_prompts(len(indices), device)
        prompt[:, 0] = labels[indices]
        loss = F.cross_entropy(model(classifier_input(as_size(render(glyphs[indices], prompt, size), 32))), prompt[:, 0])
        update(loss, model, optimizer)
    save_weights(directory, 'common_evaluator', model)
    return frozen(model)
