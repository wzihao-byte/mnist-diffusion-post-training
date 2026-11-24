"""Evaluate all checkpoints on paired prompts, or render a single prompt."""

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from diffusion import Denoiser, Diffusion
from mnist_data import digit_readout, make_reader, null_prompts, parse_prompt, prompt_text
from train import device_for, frozen, read_weights, reward


STAGES = {
    'base': 'Base',
    'sft': 'SFT',
    'rl': 'SFT + RL',
    'opd': 'SFT + OPD',
    'rl_opd': 'SFT + RL + OPD',
    'teacher': 'Teacher',
}
SEEDS = (1103, 1104)
SAMPLES_PER_PROMPT = 8
PROMPTS = list(itertools.product(range(10), range(3), range(2)))
EXAMPLES = ((0, 0, 0), (2, 1, 1), (3, 2, 0), (5, 0, 1), (7, 1, 0), (9, 2, 1))


def load_model(directory, stage, specification, device):
    width = specification['teacher_width' if stage == 'teacher' else 'student_width']
    return frozen(read_weights(directory, stage, Denoiser(width).to(device)))


@torch.no_grad()
def measure(images, prompts, evaluator, scorer):
    gray, center, shares, mass = digit_readout(images)
    digit_prediction = evaluator(gray).argmax(dim=1)
    color_share = shares.gather(1, prompts[:, 1:2]).squeeze(1)
    side_margin = (prompts[:, 2] * 2 - 1) * (center - 7.5)
    ink_ok = mass >= 4
    digit_ok = (digit_prediction == prompts[:, 0]) & ink_ok
    color_ok = (color_share >= 0.55) & ink_ok
    side_ok = (side_margin >= 0.5) & ink_ok
    return {
        'joint': digit_ok & color_ok & side_ok,
        'digit': digit_ok,
        'color': color_ok,
        'side': side_ok,
        'reward': reward(images, prompts, scorer),
        'predicted_digit': digit_prediction,
        'target_color_share': color_share,
        'side_margin_px': side_margin,
        'ink_mass': mass,
    }


def to_rgb(tensor, size):
    pixels = tensor.cpu().permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().numpy()
    return Image.fromarray(pixels).resize((size, size), Image.Resampling.NEAREST)


def font(size, bold=False):
    paths = ('/System/Library/Fonts/Supplemental/Arial{weight}.ttf',
             '/usr/share/fonts/truetype/dejavu/DejaVuSans{weight}.ttf')
    weights = (' Bold', '-Bold') if bold else ('', '')
    for path, weight in zip(paths, weights, strict=True):
        candidate = Path(path.format(weight=weight))
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default(size=size)


def comparison_image(directory, snapshots, summary):
    background, ink, muted = '#f7f8fa', '#141e2e', '#47576c'
    tile, row_pitch, column_pitch, origin = 116, 138, 145, 232
    canvas = Image.new('RGB', (origin + column_pitch * len(STAGES) + 18, 1060), background)
    draw = ImageDraw.Draw(canvas)
    draw.text((32, 24), 'Prompt-conditioned digit diffusion', fill=ink, font=font(32, True))
    draw.text((33, 68), 'Same prompts and noise for every model. First draw per prompt, with no filtering.',
              fill=muted, font=font(17))
    draw.text((33, 118), 'Prompt', fill=muted, font=font(17, True))
    for index, (stage, label) in enumerate(STAGES.items()):
        x = origin + index * column_pitch
        draw.text((x, 118), label, fill=ink, font=font(16, True))
        rate = summary[stage]['joint']
        draw.text((x, 141), f'{rate:.1%} joint pass', fill=muted, font=font(14))
    for index, prompt in enumerate(EXAMPLES):
        y = 176 + index * row_pitch
        draw.text((33, y + tile // 2 - 11), prompt_text(prompt), fill=ink, font=font(18, True))
        for column, stage in enumerate(STAGES):
            x = origin + column * column_pitch
            image, success = snapshots[stage][prompt]
            draw.rectangle((x - 2, y - 2, x + tile + 1, y + tile + 1),
                           fill='#36946d' if success else '#b6c0cf')
            canvas.paste(to_rgb(image, tile), (x, y))
    draw.text((33, 1012), 'Green border = digit, color and position passed automatic checks. Headers summarize all evaluated samples.',
              fill=muted, font=font(16))
    canvas.save(directory / 'comparison.png')


def disagreement_image(out, cases):
    if not cases:
        return
    background, ink, muted = '#f7f8fa', '#141e2e', '#47576c'
    count = min(4, len(cases))
    canvas = Image.new('RGB', (880, 340), background)
    draw = ImageDraw.Draw(canvas)
    draw.text((26, 19), 'High reward, different digit readout', fill=ink, font=font(25, True))
    draw.text((27, 56), 'SFT + RL + OPD. Highest rewards among disagreements, with distinct target digits.',
              fill=muted, font=font(15))
    for index, (row, image) in enumerate(cases[:count]):
        x = 27 + index * 212
        canvas.paste(to_rgb(image, 144), (x, 90))
        draw.text((x, 246), row['prompt'], fill=ink, font=font(15, True))
        draw.text((x, 268), f"Reward {row['reward']:.3f} | read as {row['predicted_digit']}",
                  fill=muted, font=font(14))
        draw.text((x, 290), f"Seed {row['seed']} | draw {row['draw']}", fill=muted, font=font(14))
    canvas.save(out / 'reward_disagreements.png')


def paired_interval(a, b):
    differences = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    rng = np.random.default_rng(4021)
    bootstraps = rng.choice(differences, size=(2000, len(differences)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(bootstraps, (0.025, 0.975))]


def evaluate_all(weights, specification, diffusion, device, out):
    evaluator = frozen(read_weights(weights, 'evaluator', make_reader('evaluator').to(device)))
    scorer = frozen(read_weights(weights, 'reward', make_reader('reward').to(device)))
    conditions = torch.tensor(PROMPTS, dtype=torch.long, device=device).repeat_interleave(SAMPLES_PER_PROMPT, dim=0)
    example_ids = {PROMPTS.index(prompt) * SAMPLES_PER_PROMPT: prompt for prompt in EXAMPLES}
    results, snapshots, joint_values, detailed, disagreements = {}, {}, {}, [], {}
    for stage in STAGES:
        model = load_model(weights, stage, specification, device)
        rates = {name: [] for name in ('joint', 'digit', 'color', 'side', 'reward')}
        snapshots[stage] = {}
        for seed in SEEDS:
            generator = torch.Generator().manual_seed(seed)
            for offset in range(0, len(conditions), 120):
                prompts = conditions[offset:offset + 120]
                conditioning = null_prompts(prompts) if stage == 'base' else prompts
                images, _ = diffusion.rollout(model, conditioning, generator=generator)
                scores = {key: value.cpu().tolist() for key, value in measure(images, prompts, evaluator, scorer).items()}
                for key in rates:
                    rates[key].extend(scores[key])
                for position in range(len(prompts)):
                    index = offset + position
                    prompt = PROMPTS[index // SAMPLES_PER_PROMPT]
                    row = {'stage': stage, 'seed': seed, 'draw': index % SAMPLES_PER_PROMPT,
                           'prompt': prompt_text(prompt),
                           **{key: series[position] for key, series in scores.items()}}
                    detailed.append(row)
                    if seed == SEEDS[0] and index in example_ids:
                        snapshots[stage][example_ids[index]] = (images[position].cpu(), scores['joint'][position])
                    if stage == 'rl_opd' and row['color'] and row['side'] and not row['digit'] and row['reward'] >= 0.90:
                        previous = disagreements.get(prompt[0])
                        if previous is None or row['reward'] > previous[0]['reward']:
                            disagreements[prompt[0]] = (row, images[position].cpu())
        results[stage] = {name: float(np.mean(values)) for name, values in rates.items()}
        joint_values[stage] = rates['joint']
        print(stage, json.dumps(results[stage]), flush=True)
    for stage in STAGES:
        results[stage]['joint_delta_vs_sft'] = results[stage]['joint'] - results['sft']['joint']
        results[stage]['joint_delta_vs_sft_ci95'] = paired_interval(joint_values[stage], joint_values['sft'])

    with (out / 'examples.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detailed[0]))
        writer.writeheader()
        writer.writerows(detailed)
    metrics = {
        'training_seed': specification['seed'],
        'inference_seeds': SEEDS,
        'prompt_count': len(PROMPTS),
        'images_per_stage': len(SEEDS) * len(conditions),
        'joint_rule': 'held-out OCR digit argmax, target-color share >= 0.55, side margin >= 0.5 px, ink mass >= 4',
        'stages': results,
    }
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    comparison_image(out, snapshots, results)
    disagreement_image(out, sorted(disagreements.values(), key=lambda case: case[0]['reward'], reverse=True))


def sample_prompt(weights, specification, diffusion, device, out, args):
    prompt = parse_prompt(args.prompt)
    prompts = torch.tensor(prompt, dtype=torch.long, device=device).repeat(args.count, 1)
    model = load_model(weights, args.stage, specification, device)
    conditioning = null_prompts(prompts) if args.stage == 'base' else prompts
    images, _ = diffusion.rollout(model, conditioning, generator=torch.Generator().manual_seed(args.seed))
    tile, margin = 128, 20
    columns = min(args.count, 6)
    rows = (args.count + columns - 1) // columns
    canvas = Image.new('RGB', (max(470, (tile + 8) * columns + 2 * margin),
                              (tile + 8) * rows + margin + 68), '#f7f8fa')
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), prompt_text(prompt), fill='#141e2e', font=font(22, True))
    draw.text((margin, 46), f'{STAGES[args.stage]} | seed {args.seed}', fill='#47576c', font=font(14))
    for index, image in enumerate(images):
        x = margin + (index % columns) * (tile + 8)
        y = 71 + (index // columns) * (tile + 8)
        canvas.paste(to_rgb(image, tile), (x, y))
    path = out / 'sample.png'
    canvas.save(path)
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', default='checkpoints')
    parser.add_argument('--out-dir', default='results')
    parser.add_argument('--device', default='auto', choices=('auto', 'cpu', 'mps', 'cuda'))
    parser.add_argument('--prompt', help='Generate a sample, e.g. "blue 7 on the left"')
    parser.add_argument('--stage', choices=STAGES, default='rl_opd')
    parser.add_argument('--seed', type=int, default=209)
    parser.add_argument('--count', type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.count <= 24:
        parser.error('--count must be between 1 and 24')
    weights, out = Path(args.weights), Path(args.out_dir)
    specification = json.loads((weights / 'config.json').read_text())
    device = device_for(args.device)
    diffusion = Diffusion(specification['diffusion_steps'], device)
    out.mkdir(parents=True, exist_ok=True)
    if args.prompt:
        sample_prompt(weights, specification, diffusion, device, out, args)
    else:
        evaluate_all(weights, specification, diffusion, device, out)


if __name__ == '__main__':
    main()
