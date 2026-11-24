"""Run the shape experiment checkpoints on a single prompt."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from diffusion import Denoiser, Diffusion
from evaluate import font, to_rgb
from mnist_data import parse_prompt, prompt_text
from score import LABELS, ORIGINAL
from train import device_for, frozen, read_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('runs/143'))
    parser.add_argument('--original', type=Path, default=Path('reference/143'))
    parser.add_argument('--stage', choices=LABELS, default='post_shape_low_lr16')
    parser.add_argument('--prompt', default='green 2 on the right')
    parser.add_argument('--seed', type=int, default=2711)
    parser.add_argument('--count', type=int, choices=range(1, 9), default=6)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--out', type=Path, default=Path('sample.png'))
    args = parser.parse_args()
    config = json.loads((args.run_dir / 'shape_config.json').read_text())
    size = 32 if args.stage.endswith('32') else 16
    device = device_for(args.device)
    stage, source = (ORIGINAL[args.stage], args.original) if args.stage in ORIGINAL else (args.stage, args.run_dir)
    model = frozen(read_weights(source, stage, Denoiser(config['width']).to(device)))
    prompt = parse_prompt(args.prompt)
    prompts = torch.tensor(prompt, dtype=torch.long, device=device).repeat(args.count, 1)
    diffusion = Diffusion(config['steps'], device, size)
    images, _ = diffusion.rollout(model, prompts, generator=torch.Generator().manual_seed(args.seed), paired_noise=True)
    tile = 120
    output = Image.new('RGB', (max(480, 20 + 134 * args.count), 218), '#f7f8fa')
    drawing = ImageDraw.Draw(output)
    drawing.text((16, 13), prompt_text(prompt), fill='#192738', font=font(25, True))
    drawing.text((16, 44), f'{LABELS[args.stage]} | seed {args.seed}', fill='#47576c', font=font(15))
    for index, image in enumerate(images):
        output.paste(to_rgb(image, tile), (16 + index * 134, 75))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.out)
    print(args.out)


if __name__ == '__main__':
    main()
