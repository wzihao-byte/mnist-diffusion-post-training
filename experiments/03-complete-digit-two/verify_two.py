"""Sample the trained model and verify the visible stroke ends of a handwritten two."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from complete_two import PROFILES
from diffusion import Denoiser, Diffusion
from evaluate import font, to_rgb
from mnist_data import digit_readout, make_reader, parse_prompt, prompt_text
from shape import as_size
from train import device_for, frozen, read_weights
from two_strokes import stroke_features, stroke_pass, visible_ends


def model_for(stage, run_dir, reference, config, device):
    if stage == 'base32':
        return frozen(read_weights(reference, 'rl_shape32', Denoiser(config['student_width']).to(device)))
    if stage == 'other_digits':
        return frozen(read_weights(reference, 'post_shape_low_lr16', Denoiser(config['student_width']).to(device)))
    width = config['teacher_width' if stage == 'two_teacher' else 'student_width']
    return frozen(read_weights(run_dir, stage, Denoiser(width).to(device)))


@torch.no_grad()
def acceptance(images, prompts, reader, features=None, visible=None):
    if features is None:
        features = stroke_features(images)
    if visible is None:
        visible = visible_ends(images)
    basic, x, shares, mass = digit_readout(as_size(images, 16))
    prediction = reader(basic).softmax(dim=1)[:, 2]
    color = shares.gather(1, prompts[:, 1:2]).squeeze(1)
    side = (prompts[:, 2] * 2 - 1) * (x - 7.5)
    visible_pass = torch.stack(list(visible.values())).all(dim=0)
    return stroke_pass(features) & visible_pass & (prediction >= 0.85) & (color >= 0.65) & (side >= 0.5) & (mass >= 7)


def verified_sample(model, diffusion, prompt, count, generator, reader, device, max_draws):
    accepted, total = [], 0
    while len(accepted) < count and total < max_draws:
        batch = min(12, max_draws - total)
        prompts = torch.tensor(prompt, device=device).repeat(batch, 1)
        images, _ = diffusion.rollout(model, prompts, generator=generator, paired_noise=True)
        checked = acceptance(images, prompts, reader)
        accepted.extend(images[checked].cpu())
        total += batch
    if len(accepted) < count:
        raise RuntimeError(f'{len(accepted)} of {count} requested samples passed after {total} attempts. Increase --max-draws.')
    return torch.stack(accepted[:count]), total


def save_samples(path, samples, title, tagline):
    tile, gap, margin = 132, 12, 22
    width = max(430, margin * 2 + (tile + gap) * len(samples) - gap)
    canvas = Image.new('RGB', (width, 240), '#f7f8fa')
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 15), title, fill='#172535', font=font(25, True))
    draw.text((margin, 49), tagline, fill='#455a70', font=font(15))
    for index, sample in enumerate(samples):
        canvas.paste(to_rgb(sample, tile), (margin + index * (tile + gap), 83))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('run'))
    parser.add_argument('--reference', type=Path, default=Path('reference'))
    parser.add_argument('--prompt', default='green 2 on the right')
    parser.add_argument('--stage', choices=PROFILES)
    parser.add_argument('--seed', type=int, default=2711)
    parser.add_argument('--count', type=int, choices=range(1, 9), default=6)
    parser.add_argument('--max-draws', type=int, default=72)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--out', type=Path, default=Path('sample.png'))
    parser.add_argument('--raw', action='store_true', help='Show the first draws before the stroke checks')
    args = parser.parse_args()
    config = json.loads((args.run_dir / 'complete_config.json').read_text())
    prompt = parse_prompt(args.prompt)
    device = device_for(args.device)
    if prompt[0] == 2:
        selection = args.run_dir / 'selected.json'
        stage = args.stage or (json.loads(selection.read_text())['stage'] if selection.exists() else 'two_opd')
    else:
        if args.stage:
            parser.error('Specialized checkpoints accept digit 2. Omit --stage to sample other digits.')
        stage = 'other_digits'
    size = 16 if stage == 'other_digits' else 32
    model = model_for(stage, args.run_dir, args.reference, config, device)
    diffusion = Diffusion(config['diffusion_steps'], device, size)
    generator = torch.Generator().manual_seed(args.seed)
    if prompt[0] == 2 and not args.raw:
        reader = frozen(read_weights(args.reference, 'reward', make_reader('reward').to(device)))
        samples, total = verified_sample(model, diffusion, prompt, args.count, generator, reader, device, args.max_draws)
        tagline = f'{stage} | {len(samples)} verified draws from {total} candidates'
    else:
        prompts = torch.tensor(prompt, device=device).repeat(args.count, 1)
        samples, _ = diffusion.rollout(model, prompts, generator=generator, paired_noise=True)
        tagline = f'{stage} | first {args.count} raw draws'
    save_samples(args.out, samples, prompt_text(prompt), tagline)
    print(args.out)


if __name__ == '__main__':
    main()
