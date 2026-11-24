"""Geometry scores for the digit-shape interventions."""

import torch
from torch.nn import functional as F


def as_size(images, size):
    if images.shape[-1] > size:
        return F.interpolate(images, size=(size, size), mode='area')
    if images.shape[-1] < size:
        return F.interpolate(images, size=(size, size), mode='nearest')
    return images


def geometry(images):
    gray = images.amax(dim=1).clamp(0, 1)
    ink = (gray - 0.15).clamp_min(0)
    extent = torch.arange(gray.shape[-1], dtype=gray.dtype, device=gray.device)
    y, x = torch.meshgrid(extent, extent, indexing='ij')
    mass = ink.sum(dim=(1, 2)).clamp_min(1e-6)
    center_x = (ink * x).sum(dim=(1, 2)) / mass
    center_y = (ink * y).sum(dim=(1, 2)) / mass
    dx, dy = x - center_x[:, None, None], y - center_y[:, None, None]
    slope = (ink * dx * dy).sum(dim=(1, 2)) / (ink * dy.square()).sum(dim=(1, 2)).clamp_min(1e-6)
    angle = -torch.rad2deg(torch.atan(slope))
    return gray, ink, angle, center_x


def foot_support(gray):
    """Horizontal span of the lower part of the bright glyph, relative to its height."""
    size = gray.shape[-1]
    coordinates = torch.arange(size, device=gray.device)
    threshold = (gray.amax(dim=(1, 2), keepdim=True) * 0.38).clamp_min(0.22)
    mask = gray >= threshold
    foreground_rows = mask.any(dim=-1)
    top = torch.where(foreground_rows, coordinates, size).amin(dim=1)
    bottom = torch.where(foreground_rows, coordinates, -1).amax(dim=1)
    height = (bottom - top + 1).clamp_min(1)
    lower = coordinates[None, :] >= (bottom - 0.30 * (height - 1))[:, None]
    columns = (mask & lower[:, :, None]).any(dim=1)
    first = torch.where(columns, coordinates, size).amin(dim=1)
    last = torch.where(columns, coordinates, -1).amax(dim=1)
    width = (last - first + 1).clamp_min(0)
    return width / height


def continuity(gray):
    """Fraction of foreground linked to the brightest pixel under eight-neighbor adjacency."""
    size = gray.shape[-1]
    threshold = (gray.amax(dim=(1, 2), keepdim=True) * 0.37).clamp_min(0.24)
    mask = (gray >= threshold)[:, None]
    indices = torch.arange(size * size, device=gray.device)
    peaks = gray.flatten(1).argmax(dim=1)
    connected = (indices == peaks[:, None]).reshape(-1, 1, size, size) & mask
    for _ in range(20):
        connected = (F.max_pool2d(connected.float(), 3, stride=1, padding=1) > 0) & mask
    return connected.flatten(1).sum(dim=1) / mask.flatten(1).sum(dim=1).clamp_min(1)
