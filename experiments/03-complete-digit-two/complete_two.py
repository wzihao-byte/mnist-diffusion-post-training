"""Specialize the 32 px generator on handwritten twos with a connected foot."""

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path

import torch

from ablate import inputs_at
from diffusion import Denoiser, Diffusion
from mnist_data import digit_readout, make_reader, random_prompts, render
from shape import as_size
from train import Config, device_for, fit_opd, fit_rl, fit_supervised, frozen, read_weights, reward, seed_all
from two_strokes import LIMITS, stroke_bonus, stroke_features


PROFILES = ('base32', 'two_sft', 'two_rl', 'two_teacher', 'two_opd')


def twos(count, device):
    prompts = random_prompts(count, device)
    prompts[:, 0] = 2
    return prompts


@torch.no_grad()
def complete_reward(images, prompts, reader):
    low = as_size(images, 16)
    gray, *_ = digit_readout(low)
    digit_confidence = reader(gray).softmax(dim=1)[:, 2]
    features = stroke_features(images)
    return reward(low, prompts, reader) + 0.60 * digit_confidence * stroke_bonus(features)


@torch.no_grad()
def select_training_twos(glyphs, labels):
    images = glyphs[labels == 2]
    groups = []
    for chunk in images.split(192):
        prompts = labels.new_tensor((2, 1, 0)).expand(len(chunk), 3)
        features = stroke_features(render(chunk, prompts, 32))
        stricter = {**LIMITS, 'start': 0.31, 'turn': 0.36, 'tail': 0.36,
                    'foot': 0.98, 'connected': 0.98, 'contrast': 0.80}
        selected = torch.stack([value >= stricter[key] for key, value in features.items()]).all(dim=0)
        groups.append(chunk[selected])
    kept = torch.cat(groups)
    if len(kept) < 300:
        raise ValueError(f'Insufficient complete reference glyphs: {len(kept)}')
    return (kept, labels.new_full((len(kept),), 2)), len(images)


def fit(args):
    directory, reference = Path(args.run_dir), Path(args.reference)
    directory.mkdir(parents=True, exist_ok=True)
    device = device_for(args.device)
    config = replace(Config(seed=args.seed), batch_size=72, log_interval=100, rl_steps=args.rl_steps)
    spec = {'seed': args.seed, 'size': 32, 'diffusion_steps': config.diffusion_steps,
            'sft_steps': args.sft_steps, 'rl_steps': args.rl_steps, 'teacher_steps': args.teacher_steps,
            'opd_steps': args.opd_steps, 'student_width': config.student_width, 'teacher_width': config.teacher_width,
            'device': str(device), 'torch': str(torch.__version__)}
    manifest = directory / 'complete_config.json'
    if manifest.exists():
        known = json.loads(manifest.read_text())
        if any(known.get(key) != value for key, value in spec.items()):
            raise ValueError('Use a new run directory for different training settings.')
    train_data = inputs_at(args.data_dir, args.seed, 32, device)[0]
    clean, available = select_training_twos(*train_data)
    spec.update({'source_twos': available, 'chosen_twos': len(clean[0]), 'acceptance_limits': LIMITS})
    manifest.write_text(json.dumps(spec, indent=2) + '\n')
    print(json.dumps(spec), flush=True)

    def load(name, width, source=directory):
        return read_weights(source, name, Denoiser(width).to(device))

    diffusion = Diffusion(config.diffusion_steps, device, 32)
    seed_all(args.seed + 41)
    if (directory / 'two_sft.pt').exists():
        sft = load('two_sft', config.student_width)
    else:
        sft = fit_supervised('two_sft', load('rl_shape32', config.student_width, reference), clean,
                             diffusion, directory, config, args.sft_steps, True, 'edge')
    frozen(sft)

    scorer = frozen(read_weights(reference, 'reward', make_reader('reward').to(device)))
    seed_all(args.seed + 42)
    if (directory / 'two_rl.pt').exists():
        rl = load('two_rl', config.student_width)
    else:
        rl = fit_rl(copy.deepcopy(sft).requires_grad_(True), sft, scorer, diffusion, directory, config, device,
                    stage='two_rl', reward_function=complete_reward, prompt_sampler=twos,
                    learning_rate=0.00008, kl_weight=0.16)

    seed_all(args.seed + 43)
    if (directory / 'two_teacher.pt').exists():
        teacher = load('two_teacher', config.teacher_width)
    else:
        teacher = fit_supervised('two_teacher', load('teacher', config.teacher_width, reference), clean,
                                 diffusion, directory, config, args.teacher_steps, True, 'edge')
    frozen(teacher)
    seed_all(args.seed + 44)
    if not (directory / 'two_opd.pt').exists():
        fit_opd('two_opd', rl, teacher, diffusion, directory, config, args.opd_steps, device,
                prompt_sampler=twos, learning_rate=0.00020)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default='run')
    parser.add_argument('--reference', default='reference')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--seed', type=int, default=143)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--sft-steps', type=int, default=700)
    parser.add_argument('--rl-steps', type=int, default=240)
    parser.add_argument('--teacher-steps', type=int, default=750)
    parser.add_argument('--opd-steps', type=int, default=160)
    fit(parser.parse_args())


if __name__ == '__main__':
    main()
