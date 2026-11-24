"""Compare shape interventions on matched prompts and paired Gaussian noise."""

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from ablate import RL_STAGES, SUPERVISED, inputs_at
from diffusion import Denoiser, Diffusion
from evaluate import font, to_rgb
from judge import classifier_input, train_judge
from mnist_data import prompt_text, render
from shape import as_size, foot_support, geometry
from train import device_for, frozen, read_weights


PROMPTS = list(itertools.product(range(10), range(3), range(2)))
SAMPLING_SEEDS = (2711, 2712)
REFERENCE_CUTOFF = 15.0
TILT_CUTOFF = 10.0
ORIGINAL = {'original_full': 'rl_opd'}
LABELS = {
    'original_full': 'Original full', 'sft16': '16px SFT control', 'hard2': 'Hard 2 samples',
    'foreground': 'Foreground loss', 'edge': 'Edge loss', 'sft32': '32px SFT',
    'rl_control16': '16px RL control', 'rl_shape16': '16px shape RL',
    'rl_control32': '32px RL control', 'rl_shape32': '32px shape RL',
    'post_control16': 'Original + extra RL', 'post_shape16': 'Original + shape RL',
    'post_shape_low_lr16': 'Shape RL, lower LR', 'post_shape_high_kl16': 'Shape RL, higher KL',
}
CONTROL = {**{x: 'sft16' for x in ('hard2', 'foreground', 'edge', 'sft32')},
           'rl_shape16': 'rl_control16', 'rl_shape32': 'rl_control32',
           'rl_control32': 'rl_control16', 'post_control16': 'original_full', 'post_shape16': 'post_control16',
           'post_shape_low_lr16': 'post_shape16', 'post_shape_high_kl16': 'post_shape16'}
GALLERY_PROMPTS = [(0, 0, 0), (0, 2, 1), (2, 1, 1), (2, 0, 0)]


@torch.no_grad()
def read_scores(images, prompts, judge):
    images = as_size(images, 32)
    gray, ink, angles, center = geometry(images)
    mass = ink.sum(dim=(1, 2)) / 4
    shares = images.sub(0.15).clamp_min(0).sum(dim=(2, 3))
    color = shares.gather(1, prompts[:, 1:2]).squeeze(1) / shares.sum(dim=1).clamp_min(1e-6)
    side = (prompts[:, 2] * 2 - 1) * (center - 15.5)
    prediction = judge(classifier_input(images)).argmax(dim=1)
    readable = (prediction == prompts[:, 0]) & (mass >= 4)
    joint = readable & (color >= 0.55) & (side >= 1)
    bright = (gray >= 0.50).sum(dim=(1, 2)) / 4
    return {
        'pass': joint, 'digit_ok': readable, 'prediction': prediction,
        'tilt': angles, 'upright': joint & (angles.abs() <= TILT_CUTOFF),
        'bright_area': bright, 'strong': joint & (bright >= REFERENCE_CUTOFF),
        'foot': foot_support(gray), 'ink': mass,
    }


def summary(rows):
    zeros = [r for r in rows if r['digit'] == 0]
    twos = [r for r in rows if r['digit'] == 2]
    other = [r for r in rows if r['digit'] not in (0, 2)]
    mean = lambda group, field: float(np.mean([r[field] for r in group]))
    correct_zeros = [abs(r['tilt']) for r in zeros if r['pass']]
    return {
        'images': len(rows), 'per_digit': len(zeros),
        'joint': mean(rows, 'pass'), 'other_joint': mean(other, 'pass'),
        'zero_digit': mean(zeros, 'digit_ok'), 'zero_upright': mean(zeros, 'upright'),
        'zero_abs_angle_median': float(np.median(correct_zeros)) if correct_zeros else None,
        'two_digit': mean(twos, 'digit_ok'), 'two_strong': mean(twos, 'strong'),
        'two_bright_median': float(np.median([r['bright_area'] for r in twos])),
        'two_foot_median': float(np.median([r['foot'] for r in twos])),
    }


def pair_delta(a, b):
    rng = np.random.default_rng(827)
    deltas = {}
    for label, col, digits in [('zero_upright', 'upright', [0]), ('two_digit', 'digit_ok', [2]),
                               ('two_strong', 'strong', [2]), ('other_joint', 'pass', list(set(range(10)) - {0, 2}))]:
        diff = np.asarray([int(x[col]) - int(y[col]) for x, y in zip(a, b, strict=True) if x['digit'] in digits])
        intervals = rng.choice(diff, size=(2000, len(diff)), replace=True).mean(axis=1)
        deltas[label] = [float(diff.mean()), *map(float, np.quantile(intervals, [0.025, 0.975]))]
    return deltas


def glyph_gallery(path, stages, grid, metrics, title):
    tile, pitch_x, pitch_y, start_x = 110, 150, 125, 220
    frame = Image.new('RGB', (start_x + len(stages) * pitch_x + 14, 1130), '#f7f8fa')
    drawing = ImageDraw.Draw(frame)
    drawing.text((23, 17), title, fill='#192738', font=font(27, True))
    drawing.text((23, 52), 'First two draws for each prompt | same Gaussian noise | no selection', fill='#47576c', font=font(16))
    for col, stage in enumerate(stages):
        x = start_x + col * pitch_x
        drawing.text((x, 78), LABELS[stage], fill='#192738', font=font(15, True))
        m = metrics[stage]
        text = f"0 upright {m['zero_upright']:.0%}  /  2 {m['two_strong']:.0%}"
        drawing.text((x, 99), text, fill='#47576c', font=font(12))
        for row, prompt in enumerate(GALLERY_PROMPTS):
            for draw in range(2):
                y = 123 + (row * 2 + draw) * pitch_y
                if col == 0:
                    drawing.text((23, y + 40), prompt_text(prompt), fill='#192738', font=font(16, True))
                    drawing.text((23, y + 63), f'Draw {draw + 1}', fill='#47576c', font=font(13))
                frame.paste(to_rgb(grid[stage][(prompt, draw)], tile), (x, y))
    frame.save(path)


@torch.no_grad()
def reference_check(reader, data_dir, seed, device):
    reference = {}
    for size in (16, 32):
        _, _, _, (glyphs, labels) = inputs_at(data_dir, seed, size, device)
        readings = []
        for offset in range(0, len(labels), 192):
            y = labels[offset:offset+192]
            i = torch.arange(offset, offset+len(y), device=device)
            p = torch.stack((y, i % 3, i % 2), dim=1)
            scores = read_scores(render(glyphs[offset:offset+len(y)], p, size), p, reader)
            readings.append(torch.stack((scores['digit_ok'], scores['bright_area'], scores['foot']), dim=1).cpu())
        values = torch.cat(readings)
        test_labels = labels.cpu()
        reference[size] = {'ocr': float(values[:, 0].mean()),
                           'ocr_0': float(values[test_labels == 0, 0].mean()),
                           'ocr_2': float(values[test_labels == 2, 0].mean()),
                           'bright_2_q10': float(values[test_labels == 2, 1].quantile(0.10)),
                           'bright_2_median': float(values[test_labels == 2, 1].median()),
                           'foot_2_median': float(values[test_labels == 2, 2].median())}
    return reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--repeats', type=int, default=12, help='Each prompt is sampled this often per sampling seed')
    parser.add_argument('stages', nargs='*')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.run_dir / 'shape_config.json').read_text())
    seeds, repetitions = SAMPLING_SEEDS, args.repeats
    device = device_for(args.device)
    judge = train_judge(args.data_dir, config['seed'], args.run_dir, device)
    reference = reference_check(judge, args.data_dir, config['seed'], device)
    stages = args.stages or tuple(stage for stage in LABELS if stage in ORIGINAL or (args.run_dir / f'{stage}.pt').exists())
    conditions = torch.tensor(PROMPTS, device=device).repeat_interleave(repetitions, dim=0)
    snapshots, all_rows, metrics = {}, {}, {}
    for stage in stages:
        size = 32 if stage.endswith('32') else 16
        source = args.original if stage in ORIGINAL else args.run_dir
        model = frozen(read_weights(source, ORIGINAL.get(stage, stage), Denoiser(config['width']).to(device)))
        diffusion = Diffusion(config['steps'], device, size)
        rows, snapshots[stage] = [], {}
        for seed in seeds:
            generator = torch.Generator().manual_seed(seed)
            for offset in range(0, len(conditions), 96):
                prompts = conditions[offset:offset+96]
                images, _ = diffusion.rollout(model, prompts, generator=generator, paired_noise=True)
                values = {key: value.cpu().tolist() for key, value in read_scores(images, prompts, judge).items()}
                for position in range(len(prompts)):
                    i = offset + position
                    prompt, draw = PROMPTS[i // repetitions], i % repetitions
                    rows.append({'training_seed': config['seed'], 'stage': stage, 'sampling_seed': seed, 'draw': draw,
                                 'digit': prompt[0], 'prompt': prompt_text(prompt),
                                 **{key: items[position] for key, items in values.items()}})
                    if seed == seeds[0] and draw < 2 and prompt in GALLERY_PROMPTS:
                        snapshots[stage][(prompt, draw)] = images[position].cpu()
        all_rows[stage] = rows
        metrics[stage] = summary(rows)
        print(stage, json.dumps(metrics[stage]), flush=True)
    tests = {stage: {'control': control, 'effect_and_95_ci': pair_delta(all_rows[stage], all_rows[control])}
             for stage, control in CONTROL.items() if stage in metrics and control in metrics}
    report = {'seed': config['seed'], 'sampling_seeds': seeds, 'repeat_per_seed': repetitions,
              'tilt_cutoff_deg': TILT_CUTOFF, 'brightness_cutoff_area16': REFERENCE_CUTOFF,
              'reference': reference, 'stages': metrics, 'paired_tests': tests}
    (args.out / 'shape_metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    with (args.out / 'shape_samples.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(all_rows.values()))[0]))
        writer.writeheader()
        for rows in all_rows.values():
            writer.writerows(rows)
    for name, group, title in (
        ('supervised', SUPERVISED, 'Shape ablation | supervised fine-tuning'),
        ('rl', RL_STAGES, 'Shape ablation | reward comparisons'),
        ('final', ('original_full', 'post_control16', 'post_shape16', 'post_shape_low_lr16', 'post_shape_high_kl16'),
         'Original pipeline | with and without a shape reward'),
    ):
        present = [stage for stage in group if stage in metrics]
        if len(present) >= 2:
            glyph_gallery(args.out / f'{name}.png', present, snapshots, metrics, title)


if __name__ == '__main__':
    main()
