"""Exercise the default sampling and rechecking path on all six prompts."""

import argparse
import json
from pathlib import Path

import torch

from audit_two import CONDITIONS, score_batch
from diffusion import Diffusion
from judge import new_reader
from mnist_data import make_reader, prompt_text
from train import device_for, frozen, read_weights
from verify_two import model_for, verified_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('run'))
    parser.add_argument('--reference', type=Path, default=Path('reference'))
    parser.add_argument('--out', type=Path, default=Path('delivery.json'))
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    device = device_for(args.device)
    config = json.loads((args.run_dir / 'complete_config.json').read_text())
    stage = json.loads((args.run_dir / 'selected.json').read_text())['stage']
    model = model_for(stage, args.run_dir, args.reference, config, device)
    diffusion = Diffusion(config['diffusion_steps'], device, 32)
    verifier = frozen(read_weights(args.reference, 'reward', make_reader('reward').to(device)))
    judge = frozen(read_weights(args.reference, 'common_evaluator', new_reader().to(device)))
    rows = []
    for prompt in CONDITIONS:
        for seed in (301, 302, 303, 304):
            images, total = verified_sample(model, diffusion, prompt, 4,
                                            torch.Generator().manual_seed(seed), verifier, device, 48)
            prompts = torch.tensor(prompt, device=device).repeat(len(images), 1)
            result = score_batch(images.to(device), prompts, judge, verifier)
            assert result['checked'].all(), 'Returned output failed the stroke check'
            rows.append({'prompt': prompt_text(prompt), 'seed': seed, 'delivered': len(images),
                         'batch_candidates': total, 'independently_read_as_2': int(result['readable'].sum())})
    report = {'stage': stage, 'requests': len(rows), 'delivered': sum(r['delivered'] for r in rows),
              'independently_read_as_2': sum(r['independently_read_as_2'] for r in rows), 'records': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(report['requests'], report['delivered'], report['independently_read_as_2'])


if __name__ == '__main__':
    main()
