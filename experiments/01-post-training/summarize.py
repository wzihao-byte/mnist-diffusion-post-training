"""Aggregate joint pass rates from independent training runs."""

import argparse
import json
import statistics
from pathlib import Path


def summarize(inputs):
    reports = [json.loads(path.read_text()) for path in inputs]
    seeds = [report['training_seed'] for report in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Training seeds must be distinct.')

    reference = reports[0]
    for report in reports[1:]:
        for key in ('inference_seeds', 'images_per_stage', 'prompt_count', 'joint_rule'):
            if report[key] != reference[key]:
                raise ValueError(f'Incompatible evaluation protocol: {key}')
        if report['stages'].keys() != reference['stages'].keys():
            raise ValueError('Each report must contain the same stages.')

    stages = {}
    for stage in reference['stages']:
        values = [report['stages'][stage]['joint'] for report in reports]
        deltas = [report['stages'][stage]['joint_delta_vs_sft'] for report in reports]
        stages[stage] = {
            'joint_by_seed': dict(zip(seeds, values, strict=True)),
            'mean_joint': statistics.mean(values),
            'sd_joint': statistics.stdev(values) if len(values) > 1 else None,
            'mean_delta_vs_sft': statistics.mean(deltas),
        }
    return {'training_seeds': seeds, 'images_per_stage_per_seed': reference['images_per_stage'], 'stages': stages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, default=Path('multi_seed.json'))
    args = parser.parse_args()
    report = summarize(args.reports)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    for name, values in report['stages'].items():
        print(f"{name:8s} {values['mean_joint']:.2%} | delta vs SFT {100 * values['mean_delta_vs_sft']:+.2f} pp")


if __name__ == '__main__':
    main()
