"""Summarize the matched changes across experiment seeds."""

import argparse
import json
import statistics
from pathlib import Path


FIELDS = ('joint', 'other_joint', 'zero_upright', 'zero_abs_angle_median', 'two_digit',
          'two_strong', 'two_bright_median', 'two_foot_median')


def build(reports):
    seeds = [report['seed'] for report in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Each training seed may be included once.')
    for report in reports[1:]:
        for field in ('sampling_seeds', 'repeat_per_seed', 'tilt_cutoff_deg', 'brightness_cutoff_area16'):
            if reports[0][field] != report[field]:
                raise ValueError(f'Mismatched evaluation setting: {field}')
    all_stages = dict.fromkeys(stage for report in reports for stage in report['stages'])
    output = {}
    for stage in all_stages:
        raw = {str(r['seed']): r['stages'][stage] for r in reports if stage in r['stages']}
        values = {field: statistics.mean([metrics[field] for metrics in raw.values()]) for field in FIELDS}
        control = None
        effects = {}
        for report in reports:
            if stage not in report['paired_tests']:
                continue
            entry = report['paired_tests'][stage]
            if control is not None and control != entry['control']:
                raise ValueError(f'Conflicting controls for {stage}.')
            control = entry['control']
            effects[str(report['seed'])] = entry['effect_and_95_ci']
        means = {field: statistics.mean([diff[field][0] for diff in effects.values()])
                 for field in next(iter(effects.values()), {})}
        output[stage] = {'means': values, 'by_seed': raw, 'paired_control': control,
                         'delta_mean': means, 'delta_by_seed': effects}
    return {'seeds': seeds, 'stages': output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = build([json.loads(path.read_text()) for path in args.reports])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    for stage, row in result['stages'].items():
        m = row['means']
        print(f"{stage:20} n={len(row['by_seed'])} 0 upright={m['zero_upright']:.2%} "
              f"2 OCR={m['two_digit']:.2%} 2 strong={m['two_strong']:.2%} other={m['other_joint']:.2%}")


if __name__ == '__main__':
    main()
