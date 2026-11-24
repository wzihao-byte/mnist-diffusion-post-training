"""Measure the connected start, turn and foot of the intended handwritten 2 style."""

import torch
from torch.nn import functional as F


LIMITS = {'start': 0.26, 'turn': 0.28, 'tail': 0.30, 'foot': 0.90, 'connected': 0.94, 'contrast': 0.70}


VISIBLE_LIMITS = {'visible_start': 0.19, 'visible_turn': 0.25, 'visible_tail': 0.27}


def halo_seed(gray, mask):
    size = gray.shape[-1]
    peak = gray.flatten(1).argmax(dim=1)
    index = torch.arange(size * size, device=gray.device)
    return (index == peak[:, None]).reshape(-1, 1, size, size) & mask[:, None]


def connected_ink(gray, mask):
    connected = halo_seed(gray, mask)
    for _ in range(gray.shape[-1] + 8):
        connected = (F.max_pool2d(connected.float(), 3, stride=1, padding=1) > 0) & mask[:, None]
    return connected[:, 0]


def stroke_features(images):
    """Return dimensionless measures in the visible digit's own bounding box."""
    gray = images.amax(dim=1).clamp(0, 1)
    brightness = gray.amax(dim=(1, 2))
    mask = gray >= (brightness[:, None, None] * 0.43).clamp_min(0.29)
    ink = connected_ink(gray, mask)
    size = gray.shape[-1]
    coords = torch.arange(size, device=gray.device)
    columns, rows = ink.any(dim=1), ink.any(dim=2)
    left, top = [torch.where(occupied, coords, size).amin(dim=1) for occupied in (columns, rows)]
    right, bottom = [torch.where(occupied, coords, -1).amax(dim=1) for occupied in (columns, rows)]
    width, height = (right - left).clamp_min(1), (bottom - top).clamp_min(1)
    x = ((coords[None, :] - left[:, None]) / width[:, None])[:, None, :]
    y = ((coords[None, :] - top[:, None]) / height[:, None])[:, :, None]
    lower = y >= 0.68
    middle_foot = (x[:, 0, :] >= 0.19) & (x[:, 0, :] <= 0.83)
    foot_coverage = ((ink & lower).any(dim=1) & middle_foot).sum(dim=1) / middle_foot.sum(dim=1).clamp_min(1)

    def strength(region):
        return (ink & region).sum(dim=(1, 2)) / height

    return {
        'start': strength((x <= 0.37) & (y >= 0.04) & (y <= 0.42)),
        'turn': strength((x <= 0.31) & lower),
        'tail': strength((x >= 0.76) & lower),
        'foot': foot_coverage,
        'connected': ink.sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1),
        'contrast': brightness,
    }


def stroke_pass(features):
    return torch.stack([features[key] >= bound for key, bound in LIMITS.items()]).all(dim=0)


def stroke_bonus(features):
    scores = torch.stack([(features[key] / bound).clamp(0, 1.25) / 1.25 for key, bound in LIMITS.items()])
    return 0.55 * scores.amin(dim=0) + 0.45 * scores.mean(dim=0)


def visible_ends(images):
    gray = images.amax(dim=1)
    body = gray >= (gray.amax(dim=(1, 2), keepdim=True) * 0.38).clamp_min(0.28)
    size = gray.shape[-1]
    coords = torch.arange(size, device=gray.device)
    left, top = [torch.where(occupied, coords, size).amin(dim=1) for occupied in (body.any(dim=1), body.any(dim=2))]
    right, bottom = [torch.where(occupied, coords, -1).amax(dim=1) for occupied in (body.any(dim=1), body.any(dim=2))]
    width, height = (right - left).clamp_min(1), (bottom - top).clamp_min(1)
    x = ((coords[None, :] - left[:, None]) / width[:, None])[:, None, :]
    y = ((coords[None, :] - top[:, None]) / height[:, None])[:, :, None]
    bright = gray >= 0.50

    def mass(region):
        return (bright & region).sum(dim=(1, 2)) / height

    edge = (coords == 0) | (coords == size - 1)
    boundary = edge[:, None] | edge[None, :]
    background = ~body[:, None]
    outside = boundary[None, None] & background
    for _ in range(size + 8):
        horizontal = F.max_pool2d(outside.float(), (1, 3), stride=1, padding=(0, 1))
        vertical = F.max_pool2d(outside.float(), (3, 1), stride=1, padding=(1, 0))
        outside = ((horizontal + vertical) > 0) & background
    holes = (background & ~outside)[:, 0]
    upper_hole = (holes & (y <= 0.68)).sum(dim=(1, 2))
    lower_hole = (holes & (y > 0.68)).sum(dim=(1, 2))
    upper_columns = (bright & (y <= 0.46)).any(dim=1)
    lower_columns = (bright & (y >= 0.68)).any(dim=1)
    upper_right = torch.where(upper_columns, coords, -1).amax(dim=1)
    lower_right = torch.where(lower_columns, coords, -1).amax(dim=1)
    tail_reach = (lower_right >= upper_right - 1) & (lower_right >= left + 0.82 * width)
    return {
        'visible_start': mass((x <= 0.37) & (y >= 0.04) & (y <= 0.42)) >= VISIBLE_LIMITS['visible_start'],
        'visible_turn': mass((x <= 0.31) & (y >= 0.68)) >= VISIBLE_LIMITS['visible_turn'],
        'visible_tail': mass((x >= 0.76) & (y >= 0.68)) >= VISIBLE_LIMITS['visible_tail'],
        'open_hook': upper_hole <= 1,
        'open_foot': lower_hole <= 1,
        'tail_reach': tail_reach,
        'full_frame': (gray * boundary).amax(dim=(1, 2)) < 0.29,
    }
