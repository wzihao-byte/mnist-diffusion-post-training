"""Train SFT, reward-guided DDPM and on-policy distillation on colored MNIST."""

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from torch.nn import functional as F

from diffusion import Denoiser, Diffusion
from mnist_data import batch_examples, digit_readout, make_reader, null_prompts, random_prompts, read_mnist, render


@dataclass(frozen=True)
class Config:
    seed: int = 143
    diffusion_steps: int = 16
    student_width: int = 16
    teacher_width: int = 32
    batch_size: int = 96
    reader_steps: int = 450
    base_steps: int = 850
    sft_steps: int = 1050
    teacher_steps: int = 2700
    rl_steps: int = 180
    opd_steps: int = 215
    chain_steps: int = 215
    rollout_batch: int = 32
    reward_group: int = 4
    log_interval: int = 50


def device_for(request):
    if request == 'auto':
        request = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.device(request)
    if device.type == 'cpu':
        torch.set_num_threads(4)
    return device


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def record(directory, stage, step, started, **values):
    row = {'stage': stage, 'step': step, 'seconds': round(time.monotonic() - started, 2), **values}
    line = json.dumps(row, allow_nan=False)
    with (directory / 'training.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(line + '\n')
    print(line, flush=True)


def save_weights(directory, stage, model):
    target = directory / f'{stage}.pt'
    temporary = target.with_suffix('.tmp')
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, temporary)
    temporary.replace(target)


def read_weights(directory, stage, model):
    model.load_state_dict(torch.load(directory / f'{stage}.pt', map_location='cpu', weights_only=True))
    return model


def frozen(model):
    return model.eval().requires_grad_(False)


def update(loss, model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return norm


def train_reader(kind, data, test_data, directory, config, device):
    model = make_reader(kind).to(device)
    if (directory / f'{kind}.pt').exists():
        return frozen(read_weights(directory, kind, model))
    start = time.monotonic()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    for step in range(1, config.reader_steps + 1):
        images, prompts = batch_examples(*data, config.batch_size)
        gray, *_ = digit_readout(images.add(1).mul(0.5))
        loss = F.cross_entropy(model(gray), prompts[:, 0])
        update(loss, model, optimizer)
        if step == 1 or step % config.log_interval == 0:
            record(directory, kind, step, start, loss=loss.item())
    frozen(model)
    with torch.no_grad():
        hits = 0
        images, labels = test_data
        for offset in range(0, len(images), 256):
            slice_ = slice(offset, offset + 256)
            image_slice, label_slice = images[slice_], labels[slice_]
            offsets = torch.arange(offset, offset + len(image_slice), device=device)
            prompts = torch.stack((label_slice, offsets % 3, offsets % 2), dim=1)
            gray, *_ = digit_readout(render(image_slice, prompts))
            hits += (model(gray).argmax(dim=1) == label_slice).sum().item()
    record(directory, kind, config.reader_steps, start, mnist_test_accuracy=hits / len(images))
    save_weights(directory, kind, model)
    return model


def fit_supervised(stage, model, data, diffusion, directory, config, iterations, conditioned,
                   objective='mse', replacements=None):
    start = time.monotonic()
    model.train()
    ema = frozen(copy.deepcopy(model))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0013, weight_decay=0.01)
    for step in range(1, iterations + 1):
        clean, prompts = batch_examples(*data, config.batch_size, diffusion.image_size, replacements)
        if not conditioned:
            prompts = null_prompts(prompts)
        loss = diffusion.supervised_loss(model, clean, prompts, objective)
        update(loss, model, optimizer)
        with torch.no_grad():
            for average, current in zip(ema.parameters(), model.parameters(), strict=True):
                average.lerp_(current, 1.0 - min(0.99, (step - 1) / (step + 5)))
        if step == 1 or step % config.log_interval == 0 or step == iterations:
            record(directory, stage, step, start, loss=loss.item())
    save_weights(directory, stage, ema)
    return ema


@torch.no_grad()
def reward(images, prompts, reader):
    gray, x_center, colors, mass = digit_readout(images)
    digit_p = reader(gray).softmax(dim=1).gather(1, prompts[:, :1]).squeeze(1)
    color_p = colors.gather(1, prompts[:, 1:2]).squeeze(1)
    side = (prompts[:, 2] * 2 - 1) * (x_center - 7.5)
    correct_side = (side * 1.4).sigmoid()
    background_penalty = ((mass - 44) / 18).clamp(0, 1) + ((4 - mass) / 4).clamp(0, 1)
    return 0.54 * digit_p + 0.27 * color_p + 0.19 * correct_side - 0.18 * background_penalty


def fit_rl(model, reference, scorer, diffusion, directory, config, device, *, stage='rl',
           reward_function=reward, target_prompts=False, learning_rate=0.00016, kl_weight=0.10,
           prompt_sampler=random_prompts):
    start = time.monotonic()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    group = config.reward_group
    for step in range(1, config.rl_steps + 1):
        prompts = prompt_sampler(config.rollout_batch // group, device)
        if target_prompts:
            prompts[:4, 0] = prompts.new_tensor((0, 0, 2, 2))
        prompts = prompts.repeat_interleave(group, dim=0)
        images, path = diffusion.rollout(model, prompts, trajectory=True)
        terminal = reward_function(images, prompts, scorer).reshape(-1, group)
        # Baseline uses the other draws for the same prompt, independent of this draw.
        baseline = (terminal.sum(dim=1, keepdim=True) - terminal) / (group - 1)
        advantage = (terminal - baseline).flatten()
        advantage = advantage / advantage.std().clamp_min(0.05)
        state, times, indices = diffusion.sample_states(path, places=3, skip_deterministic=True)
        action = path[times - 1, indices]
        selected = prompts[indices]
        variance = diffusion.value(diffusion.variance, times)
        mean = diffusion.mean(state, times, model(state, times, selected))
        # Constants in the Gaussian log density are independent of the policy.
        log_prob = (-0.5 * (action - mean).square() / variance).flatten(1).mean(dim=1)
        with torch.no_grad():
            reference_mean = diffusion.mean(state, times, reference(state, times, selected))
        kl = (0.5 * (mean - reference_mean).square() / variance).mean()
        policy_loss = -(advantage[indices] * log_prob).mean()
        loss = policy_loss + kl_weight * kl
        update(loss, model, optimizer)
        if step == 1 or step % 20 == 0 or step == config.rl_steps:
            record(directory, stage, step, start, loss=loss.item(), kl=kl.item(),
                   rollout_reward=terminal.mean().item(), reward_std=terminal.std().item())
    save_weights(directory, stage, model)
    return model


def fit_opd(stage, student, teacher, diffusion, directory, config, iterations, device, *,
            prompt_sampler=random_prompts, learning_rate=0.00045):
    start = time.monotonic()
    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate, weight_decay=0.01)
    for step in range(1, iterations + 1):
        prompts = prompt_sampler(config.rollout_batch, device)
        _, path = diffusion.rollout(student, prompts, trajectory=True)
        state, times, indices = diffusion.sample_states(path, places=3)
        selected = prompts[indices]
        with torch.no_grad():
            target = teacher(state, times, selected)
        # Match the teacher's clean-image prediction on the student's visited states.
        # At fixed t this is proportional to Gaussian mean matching before clipping.
        loss = F.mse_loss(student(state, times, selected), target)
        update(loss, student, optimizer)
        if step == 1 or step % 20 == 0 or step == iterations:
            record(directory, stage, step, start, teacher_x0_mse=loss.item())
    save_weights(directory, stage, student)
    return student


def run(args, config):
    device = device_for(args.device)
    directory = Path(args.run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = directory / 'config.json'
    specification = {**asdict(config), 'device': str(device), 'torch': str(torch.__version__), 'python': sys.version.split()[0]}
    if metadata.exists() and json.loads(metadata.read_text()) != specification:
        raise ValueError('Run settings differ. Pick a new --run-dir for a different configuration.')
    metadata.write_text(json.dumps(specification, indent=2) + '\n')
    print(f'Training on {device}', flush=True)
    data, test_data = read_mnist(args.data_dir, device)
    split = torch.randperm(len(data[1]), generator=torch.Generator().manual_seed(config.seed)).to(device)
    sets = [tuple(part[indices] for part in data) for indices in split.split((48_000, 6_000, 6_000))]
    gen_data, reward_data, evaluator_data = sets
    diffusion = Diffusion(config.diffusion_steps, device)

    seed_all(config.seed + 1)
    reward_reader = train_reader('reward', reward_data, test_data, directory, config, device)
    seed_all(config.seed + 2)
    train_reader('evaluator', evaluator_data, test_data, directory, config, device)

    seed_all(config.seed + 3)
    base = Denoiser(config.student_width).to(device)
    if (directory / 'base.pt').exists():
        base = read_weights(directory, 'base', base)
    else:
        base = fit_supervised('base', base, gen_data, diffusion, directory, config, config.base_steps, False)

    seed_all(config.seed + 4)
    sft = copy.deepcopy(base).requires_grad_(True)
    if (directory / 'sft.pt').exists():
        sft = read_weights(directory, 'sft', sft)
    else:
        sft = fit_supervised('sft', sft, gen_data, diffusion, directory, config, config.sft_steps, True)
    frozen(sft)

    seed_all(config.seed + 5)
    teacher = Denoiser(config.teacher_width).to(device)
    if (directory / 'teacher.pt').exists():
        teacher = read_weights(directory, 'teacher', teacher)
    else:
        teacher = fit_supervised('teacher', teacher, gen_data, diffusion, directory, config, config.teacher_steps, True)
    frozen(teacher)

    seed_all(config.seed + 6)
    rl = copy.deepcopy(sft).requires_grad_(True)
    if (directory / 'rl.pt').exists():
        rl = read_weights(directory, 'rl', rl)
    else:
        rl = fit_rl(rl, sft, reward_reader, diffusion, directory, config, device)

    seed_all(config.seed + 7)
    if not (directory / 'opd.pt').exists():
        fit_opd('opd', copy.deepcopy(sft).requires_grad_(True), teacher,
                diffusion, directory, config, config.opd_steps, device)
    seed_all(config.seed + 8)
    if not (directory / 'rl_opd.pt').exists():
        fit_opd('rl_opd', rl, teacher, diffusion, directory, config, config.chain_steps, device)
    print(f'All training stages available in {directory}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default='run')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--device', default='auto', choices=('auto', 'cpu', 'mps', 'cuda'))
    parser.add_argument('--seed', default=143, type=int)
    parser.add_argument('--smoke', action='store_true', help='Exercise every stage with two updates')
    args = parser.parse_args()
    config = Config(seed=args.seed)
    if args.smoke:
        config = replace(config, diffusion_steps=6, student_width=8, teacher_width=12, batch_size=16,
                         reader_steps=2, base_steps=2, sft_steps=2, teacher_steps=2,
                         rl_steps=2, opd_steps=2, chain_steps=2, rollout_batch=8, log_interval=1)
    if not 0 <= config.seed < 2**32:
        parser.error('--seed must be a nonnegative 32-bit integer')
    run(args, config)


if __name__ == '__main__':
    main()
