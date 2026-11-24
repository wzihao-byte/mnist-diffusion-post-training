"""Report raw and verified completion of both visible ends on balanced prompts."""

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from complete_two import PROFILES
from diffusion import Diffusion
from evaluate import font, to_rgb
from judge import classifier_input, new_reader
from mnist_data import make_reader, prompt_text
from train import device_for, frozen, read_weights
from two_strokes import LIMITS, VISIBLE_LIMITS, stroke_features, stroke_pass, visible_ends
from verify_two import acceptance, model_for


CONDITIONS = [(2, color, side) for color, side in itertools.product(range(3), range(2))]
PROMPTS = list(itertools.product(range(10), range(3), range(2)))
SEEDS = {'selection': (9112, 9113), 'final': (9911, 9912, 9913)}


def score_batch(images, prompts, judge, verifier):
    features = stroke_features(images)
    visible = visible_ends(images)
    reads_two = judge(classifier_input(images)).argmax(dim=1) == 2
    checked = acceptance(images, prompts, verifier, features, visible)
    all_regions = stroke_pass(features)
    return {'readable': reads_two, 'stroke_complete': all_regions, 'audited_complete': reads_two & all_regions,
            'checked': checked, 'checked_audit': checked & reads_two,
            **{key: value >= LIMITS[key] for key, value in features.items()}, **visible}


def summarize(rows):
    fields = [key for key in rows[0] if isinstance(rows[0][key], bool)]
    stats = {name: float(np.mean([row[name] for row in rows])) for name in fields}
    checked = sum(row['checked'] for row in rows)
    return {'generated': len(rows), 'checked_count': checked, **stats,
            'audit_when_checked': (sum(row['checked_audit'] for row in rows) / checked) if checked else None}


def draw_gallery(path, config, stages, pick, root, reference, judge, verifier, device):
    draws = 12
    conditions = torch.tensor(PROMPTS, device=device).repeat_interleave(draws, dim=0)
    groups = tuple(p * draws for p, prompt in enumerate(PROMPTS) if prompt in CONDITIONS)
    snapshots = {}
    for stage in stages:
        model = model_for(stage, root, reference, config, device)
        diffusion = Diffusion(config['diffusion_steps'], device, 32)
        generator = torch.Generator().manual_seed(2711)
        snapshots[stage] = {}
        for offset in range(0, max(groups) + draws, 96):
            prompts = conditions[offset:offset + 96]
            images, _ = diffusion.rollout(model, prompts, generator=generator, paired_noise=True)
            for group in groups:
                if offset <= group < offset + len(prompts):
                    block = images[group-offset:group-offset+draws]
                    conditions_block = prompts[group-offset:group-offset+draws]
                    measurements = score_batch(block, conditions_block, judge, verifier)
                    snapshots[stage][group] = (block.cpu(), {k: v.cpu() for k, v in measurements.items()})

    headers = ('Previous 32px', 'Complete SFT', 'Shape RL', 'Shape OPD', 'First verified output')
    order = ('base32', 'two_sft', 'two_rl', 'two_opd', pick)
    width, vertical, tile = 1010, 133, 102
    page = Image.new('RGB', (width, 121 + len(groups) * vertical + 13), '#f7f8fa')
    canvas = ImageDraw.Draw(page)
    canvas.text((22, 18), 'Handwritten 2 | both ends of the stroke', fill='#192737', font=font(28, True))
    canvas.text((23, 59), 'First raw draw across checkpoints. Last column takes the first matching output from the same sequence.',
                fill='#4b5a6d', font=font(15))
    for i, header in enumerate(headers):
        canvas.text((195 + i * 159, 92), header, fill='#192737', font=font(15, True))
    for row, group in enumerate(groups):
        y = 124 + row * vertical
        condition = PROMPTS[group // draws]
        canvas.text((23, y + 39), prompt_text(condition), fill='#192737', font=font(16, True))
        for i, key in enumerate(order):
            x = 195 + i * 159
            block, s = snapshots[key][group]
            candidates = s['checked'].nonzero().flatten()
            if i == len(order) - 1 and not len(candidates):
                canvas.text((x, y + 38), 'No verified draw', fill='#873933', font=font(14))
                continue
            first = int(candidates[0]) if i == len(order) - 1 else 0
            passed = bool(s['checked_audit'][first])
            canvas.rectangle((x-2, y-2, x+tile+1, y+tile+1), fill='#368964' if passed else '#bcc8d4')
            page.paste(to_rgb(block[first], tile), (x, y))
            if i == len(order) - 1:
                label = f'Draw {first + 1} passed'
            else:
                checks = (('visible_start', 'left start'), ('visible_turn', 'left turn'), ('visible_tail', 'right tail'),
                          ('tail_reach', 'short tail'), ('open_hook', 'closed hook'), ('open_foot', 'closed foot'), ('foot', 'base gap'),
                          ('full_frame', 'frame edge'))
                failures = [name for code, name in checks if not bool(s[code][first])]
                label = ', '.join(failures[:2]) if failures else ('Pass' if passed else 'Other check failed')
            canvas.text((x, y+108), label, fill='#47576c', font=font(12))
    page.save(path)


def run(args):
    device = device_for(args.device)
    config = json.loads((args.run_dir / 'complete_config.json').read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    judge = frozen(read_weights(args.reference, 'common_evaluator', new_reader().to(device)))
    verifier = frozen(read_weights(args.reference, 'reward', make_reader('reward').to(device)))
    stages = args.stages or tuple(stage for stage in PROFILES if stage != 'two_teacher')
    conditions = torch.tensor(CONDITIONS, device=device).repeat_interleave(args.repeats, dim=0)
    all_rows, stages_summary = [], {}
    for stage in stages:
        model = model_for(stage, args.run_dir, args.reference, config, device)
        diffusion = Diffusion(config['diffusion_steps'], device, 32)
        rows = []
        for seed in SEEDS[args.split]:
            generator = torch.Generator().manual_seed(seed)
            for offset in range(0, len(conditions), 96):
                prompts = conditions[offset:offset + 96]
                images, _ = diffusion.rollout(model, prompts, generator=generator, paired_noise=True)
                values = {key: value.cpu().tolist() for key, value in score_batch(images, prompts, judge, verifier).items()}
                for i in range(len(prompts)):
                    index = offset + i
                    rows.append({'stage': stage, 'seed': seed, 'prompt': prompt_text(CONDITIONS[index // args.repeats]),
                                 'draw': index % args.repeats, **{key: vector[i] for key, vector in values.items()}})
        stages_summary[stage] = summarize(rows)
        all_rows.extend(rows)
        print(stage, json.dumps(stages_summary[stage]), flush=True)

    meta = {'split': args.split, 'sampling_seeds': SEEDS[args.split], 'repeats': args.repeats,
            'limits': LIMITS, 'visible_limits': VISIBLE_LIMITS, 'stages': stages_summary}
    (args.out / 'audit.json').write_text(json.dumps(meta, indent=2) + '\n')
    with (args.out / 'samples.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    if args.split == 'selection' and not args.stages:
        candidates = [stage for stage in stages if stage != 'base32']
        pick = max(candidates, key=lambda stage: stages_summary[stage]['checked_audit'])
        (args.run_dir / 'selected.json').write_text(json.dumps({'stage': pick, 'selection_split': args.split}, indent=2) + '\n')
    if args.gallery:
        pick = json.loads((args.run_dir / 'selected.json').read_text())['stage']
        draw_gallery(args.out / 'comparison.png', config, stages, pick, args.run_dir, args.reference, judge, verifier, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('run'))
    parser.add_argument('--reference', type=Path, default=Path('reference'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--split', choices=SEEDS, default='final')
    parser.add_argument('--repeats', type=int, default=64)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--gallery', action='store_true')
    parser.add_argument('stages', nargs='*')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
