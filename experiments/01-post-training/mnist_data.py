"""Public MNIST loading, synthetic prompts and digit scoring."""

import gzip
import hashlib
import struct
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


MIRROR = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
FILES = {
    'train-images-idx3-ubyte.gz': 'f68b3c2dcbeaaa9fbdd348bbdeb94873',
    'train-labels-idx1-ubyte.gz': 'd53e105ee54ea40749a09fcbcd1e9432',
    't10k-images-idx3-ubyte.gz': '9fb629c4189551a2d022fa330f9573f3',
    't10k-labels-idx1-ubyte.gz': 'ec29112dd5afa0611ce80d1b7f02629c',
}
SIZE = 16
GLYPH_SIZE = 12
COLORS = ('red', 'green', 'blue')
SIDES = ('left', 'right')
NULL_PROMPT = (10, 3, 2)


def load_idx(directory, name):
    path = directory / name
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.download')
        urlretrieve(MIRROR + name, temporary)
        temporary.replace(path)
    compressed = path.read_bytes()
    if hashlib.md5(compressed, usedforsecurity=False).hexdigest() != FILES[name]:
        raise ValueError(f'MNIST checksum mismatch: {path}')
    raw = gzip.decompress(compressed)
    magic, count = struct.unpack_from('>II', raw)
    values = np.frombuffer(raw, dtype=np.uint8, offset=16 if magic == 2051 else 8).copy()
    if magic == 2051:
        return torch.from_numpy(values.reshape(count, 1, 28, 28))
    if magic != 2049 or len(values) != count:
        raise ValueError(f'Unexpected IDX header: {name}')
    return torch.from_numpy(values).long()


def read_mnist(directory, device):
    directory = Path(directory)
    pairs = []
    for split in ('train', 't10k'):
        images = load_idx(directory, f'{split}-images-idx3-ubyte.gz')
        labels = load_idx(directory, f'{split}-labels-idx1-ubyte.gz')
        glyphs = F.interpolate(images.float() / 255, size=(GLYPH_SIZE, GLYPH_SIZE), mode='area')
        pairs.append((glyphs.to(device), labels.to(device)))
    return pairs


def random_prompts(count, device):
    return torch.stack([torch.randint(limit, (count,), device=device) for limit in (10, 3, 2)], dim=1)


def null_prompts(prompts):
    return prompts.new_tensor(NULL_PROMPT).expand_as(prompts)


def render(glyphs, conditions):
    """Color a 12px glyph and paste it left or right on a 16px black canvas."""
    left = F.pad(glyphs, (0, 4, 2, 2))
    right = F.pad(glyphs, (4, 0, 2, 2))
    canvas = torch.where(conditions[:, 2, None, None, None] == 0, left, right)
    channels = (conditions[:, 1, None] == torch.arange(3, device=glyphs.device)).float()
    return canvas * channels[:, :, None, None]


def batch_examples(glyphs, labels, count):
    indices = torch.randint(len(glyphs), (count,), device=glyphs.device)
    prompts = random_prompts(count, glyphs.device)
    prompts[:, 0] = labels[indices]
    return render(glyphs[indices], prompts).mul(2).sub(1), prompts


def digit_readout(images):
    """Score color and location on the full frame, center a grayscale copy for OCR."""
    ink = (images.clamp(0, 1) - 0.15).clamp_min(0)
    gray = ink.amax(dim=1, keepdim=True)
    grid = torch.arange(SIZE, device=images.device, dtype=images.dtype)
    mass = gray.sum(dim=(1, 2, 3)).clamp_min(1e-6)
    x_center = (gray * grid).sum(dim=(1, 2, 3)) / mass
    shift = (x_center - (SIZE - 1) / 2).round().long()
    source = grid.long()[None, :] + shift[:, None]
    indices = source.clamp(0, SIZE - 1)[:, None, None, :].expand_as(gray)
    mask = ((source >= 0) & (source < SIZE))[:, None, None, :]
    centered = gray.gather(-1, indices).mul(mask).div(0.85)
    color_mass = ink.sum(dim=(2, 3))
    shares = color_mass / color_mass.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return centered, x_center, shares, mass


def make_reader(kind):
    if kind == 'reward':
        return nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.SiLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(48 * 4 * 4, 96), nn.SiLU(), nn.Linear(96, 10),
        )
    return nn.Sequential(
        nn.Flatten(), nn.Linear(SIZE * SIZE, 256), nn.GELU(),
        nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 10),
    )


def prompt_text(prompt):
    digit, color, side = map(int, prompt)
    return f'{COLORS[color]} {digit} on the {SIDES[side]}'


def parse_prompt(text):
    # A deliberately small grammar, e.g. "red 7 on the left".
    words = text.lower().split()
    if (len(words) != 5 or words[0] not in COLORS or words[1] not in '0123456789'
            or len(words[1]) != 1 or words[2:4] != ['on', 'the'] or words[4] not in SIDES):
        raise ValueError('Use a prompt like "red 7 on the left" (red/green/blue, 0–9, left/right).')
    return int(words[1]), COLORS.index(words[0]), SIDES.index(words[4])
