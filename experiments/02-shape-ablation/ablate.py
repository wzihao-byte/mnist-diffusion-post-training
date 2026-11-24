"""Train controlled variants of the MNIST digit diffusion model."""

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path

import torch

from diffusion import Denoiser, Diffusion
from mnist_data import digit_readout, make_reader, read_mnist, render
from shape import as_size, continuity, foot_support, geometry
from train import Config, device_for, fit_rl, fit_supervised, frozen, read_weights, reward, seed_all


SUPERVISED = ('sft16', 'hard2', 'foreground', 'edge', 'sft32')
RL_STAGES = ('rl_control16', 'rl_shape16', 'rl_control32', 'rl_shape32')
POST_STAGES = ('post_control16', 'post_shape16', 'post_shape_low_lr16', 'post_shape_high_kl16')


def inputs_at(data_dir, seed, size, device):
    full, test = read_mnist(data_dir, device, glyph_size=size * 3 // 4)
    indices = torch.randperm(len(full[1]), generator=torch.Generator().manual_seed(seed)).to(device)
    train, reward_set, evaluator = [tuple(part[i] for part in full) for i in indices.split((48000, 6000, 6000))]
    return train, reward_set, evaluator, test


@torch.no_grad()
def difficult_twos(data, scorer, seed):
    """Replace half of the original '2' indices with hard real '2' glyphs."""
    glyphs, labels = data
    ids = (labels == 2).nonzero(as_tuple=True)[0]
    batches = []
    for chunk in ids.split(256):
        prompt = chunk.new_tensor((2, 1, 1)).expand(len(chunk), 3)
        gray, *_ = digit_readout(render(glyphs[chunk], prompt))
        batches.append(scorer(gray).softmax(dim=1)[:, 2])
    scores = torch.cat(batches)
    hard = ids[scores.argsort()[:len(ids) // 4]]
    rng = torch.Generator().manual_seed(seed + 90)
    selected = torch.randperm(len(ids), generator=rng)[:len(ids) // 2].to(ids.device)
    chosen = torch.randint(len(hard), (len(selected),), generator=rng).to(ids.device)
    mapping = torch.arange(len(labels), device=labels.device)
    mapping[ids[selected]] = hard[chosen]
    print(json.dumps({'hard_two_pool': len(hard), 'two_source_count': len(ids),
                      'hard_probability_mean': float(scores.sort().values[:len(hard)].mean()),
                      'all_probability_mean': float(scores.mean())}), flush=True)
    return mapping


def baseline_reward(images, prompts, scorer):
    return reward(as_size(images, 16), prompts, scorer)


@torch.no_grad()
def shape_reward(images, prompts, scorer):
    canonical = as_size(images, 16)
    base = reward(canonical, prompts, scorer)
    gray_for_reader, *_ = digit_readout(canonical)
    digit_gate = scorer(gray_for_reader).softmax(dim=1).gather(1, prompts[:, :1]).squeeze(1)
    gray, ink, angles, _ = geometry(canonical)
    has_ink = (ink.sum(dim=(1, 2)) / 9).clamp(0, 1)
    upright = torch.exp(-angles.square() / 10 ** 2)
    two_quality = continuity(gray) * (foot_support(gray) / 0.75).clamp(0, 1) * has_ink
    bonus = torch.where(prompts[:, 0] == 0, 0.40 * upright * has_ink, 0)
    bonus = bonus + torch.where(prompts[:, 0] == 2, 0.34 * two_quality, 0)
    return base + digit_gate * bonus


def train(args):
    device = device_for(args.device)
    run_dir, original = Path(args.run_dir), Path(args.original)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = replace(Config(seed=args.seed), log_interval=100, rl_steps=args.rl_steps, teacher_steps=1450)
    manifest = {'seed': config.seed, 'torch': str(torch.__version__), 'device': str(device),
                'steps': config.diffusion_steps, 'width': config.student_width,
                'teacher_width': config.teacher_width, 'rl_steps': config.rl_steps,
                'base_steps': config.base_steps, 'sft_steps': config.sft_steps, 'teacher_steps': config.teacher_steps}
    path = run_dir / 'shape_config.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Choose a different run directory when changing the training configuration.')
    path.write_text(json.dumps(manifest, indent=2) + '\n')

    variants = args.stages or (*SUPERVISED, *RL_STAGES)
    data16 = data32 = scorer = None
    if any(stage in (*SUPERVISED[:-1], *RL_STAGES, *POST_STAGES) for stage in variants):
        scorer = frozen(read_weights(original, 'reward', make_reader('reward').to(device)))
    if any(stage in SUPERVISED[:-1] for stage in variants):
        data16 = inputs_at(args.data_dir, config.seed, 16, device)[0]
    if 'sft32' in variants:
        data32 = inputs_at(args.data_dir, config.seed, 32, device)[0]

    def existing(stage, width=config.student_width, directory=run_dir):
        return read_weights(directory, stage, Denoiser(width).to(device))

    base16 = None
    if any(stage in SUPERVISED[:-1] for stage in variants):
        base16 = existing('base', directory=original)
    for stage in variants:
        if (run_dir / f'{stage}.pt').exists():
            continue
        size = 32 if stage.endswith('32') else 16
        diffusion = Diffusion(config.diffusion_steps, device, size)
        if stage in SUPERVISED:
            if size == 32:
                seed_all(config.seed + 3)
                if (run_dir / 'base32.pt').exists():
                    base = existing('base32')
                else:
                    base = fit_supervised('base32', Denoiser(config.student_width).to(device), data32,
                                          diffusion, run_dir, config, config.base_steps, False)
            else:
                base = base16
            seed_all(config.seed + 4)
            mapping = difficult_twos(data16, scorer, config.seed) if stage == 'hard2' else None
            criterion = stage if stage in ('foreground', 'edge') else 'mse'
            fit_supervised(stage, copy.deepcopy(base).requires_grad_(True), data32 if size == 32 else data16,
                           diffusion, run_dir, config, config.sft_steps, True, criterion, mapping)
        elif stage in (*RL_STAGES, *POST_STAGES):
            is_post = stage in POST_STAGES
            seed_all(config.seed + (19 if is_post else 6))
            source = frozen(existing('rl_opd', directory=original) if is_post else existing(f'sft{size}'))
            fn = shape_reward if 'shape' in stage else baseline_reward
            fit_rl(copy.deepcopy(source).requires_grad_(True), source, scorer, diffusion, run_dir, config, device,
                   stage=stage, reward_function=fn, target_prompts=True,
                   learning_rate=0.00008 if 'low_lr' in stage else 0.00016,
                   kl_weight=0.30 if 'high_kl' in stage else 0.10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', required=True, help='Original project checkpoints for the same seed')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--seed', type=int, default=143)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--rl-steps', type=int, default=360)
    parser.add_argument('stages', nargs='*')
    args = parser.parse_args()
    allowed = (*SUPERVISED, *RL_STAGES, *POST_STAGES)
    if any(stage not in allowed for stage in args.stages):
        parser.error('Choose stages from ' + ', '.join(allowed))
    train(args)


if __name__ == '__main__':
    main()
