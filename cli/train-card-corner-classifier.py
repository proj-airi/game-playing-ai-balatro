#!/usr/bin/env python3
"""Train a small Balatro card-corner rank/suit classifier."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_cli_dir = Path(__file__).parent
_src_dir = _cli_dir.parent / 'src'
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from ai_balatro.datasets.card_corner_classifier import (  # noqa: E402
    CardCornerClassifier,
    CardCornerDataset,
    class_weights,
    evaluate,
    load_labeled_rows,
    split_rows,
)

_repo_root = _cli_dir.parent
DEFAULT_RUNS_ROOT = _repo_root / 'runs' / 'classify' / 'card-corner-cnn'


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Train card-corner rank/suit CNN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--image-root', required=True, type=Path)
    parser.add_argument(
        '--out-dir',
        default=None,
        type=Path,
        help='Output directory. Defaults to runs/classify/card-corner-cnn/<run-name>.',
    )
    parser.add_argument(
        '--run-name',
        default=None,
        help='Run directory name when --out-dir is not provided.',
    )
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--learning-rate', default=1e-3, type=float)
    parser.add_argument('--image-size', default=64, type=int)
    parser.add_argument('--validation-fraction', default=0.2, type=float)
    parser.add_argument('--seed', default=7, type=int)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--trim-ink', action='store_true')
    parser.add_argument('--weighted-loss', action='store_true')
    parser.add_argument(
        '--device',
        default='auto',
        choices=['auto', 'cpu', 'mps', 'cuda'],
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = _select_device(args.device)
    out_dir = _resolve_output_dir(args)

    rows = load_labeled_rows(args.manifest)
    train_rows, validation_rows = split_rows(
        rows,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    train_dataset = CardCornerDataset(
        rows=train_rows,
        image_root=args.image_root,
        image_size=args.image_size,
        augment=args.augment,
        trim_ink=args.trim_ink,
    )
    validation_dataset = CardCornerDataset(
        rows=validation_rows,
        image_root=args.image_root,
        image_size=args.image_size,
        trim_ink=args.trim_ink,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = CardCornerClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    rank_weight = (
        class_weights(train_rows, label='rank').to(device)
        if args.weighted_loss
        else None
    )
    suit_weight = (
        class_weights(train_rows, label='suit').to(device)
        if args.weighted_loss
        else None
    )

    history = []
    best_metrics = None
    best_state_dict = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for images, ranks, suits in train_loader:
            images = images.to(device)
            ranks = ranks.to(device)
            suits = suits.to(device)
            optimizer.zero_grad(set_to_none=True)
            rank_logits, suit_logits = model(images)
            loss = F.cross_entropy(
                rank_logits, ranks, weight=rank_weight
            ) + F.cross_entropy(
                suit_logits,
                suits,
                weight=suit_weight,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * ranks.numel()
            total += ranks.numel()

        metrics = evaluate(model, validation_loader, device=device)
        metrics['epoch'] = epoch
        metrics['train_loss'] = total_loss / max(1, total)
        history.append(metrics)
        if best_metrics is None or _is_better(metrics, best_metrics):
            best_metrics = metrics.copy()
            best_state_dict = copy.deepcopy(model.state_dict())
        print(
            f'epoch={epoch:03d} loss={metrics["train_loss"]:.4f} '
            f'rank_acc={metrics["rank_accuracy"]:.3f} '
            f'suit_acc={metrics["suit_accuracy"]:.3f} '
            f'exact_acc={metrics["exact_accuracy"]:.3f}'
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / 'card-corner-classifier.pt'
    best_checkpoint_path = out_dir / 'card-corner-classifier-best.pt'
    metrics_path = out_dir / 'metrics.json'
    args_path = out_dir / 'args.json'
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'image_size': args.image_size,
            'history': history,
        },
        checkpoint_path,
    )
    torch.save(
        {
            'model_state_dict': best_state_dict or model.state_dict(),
            'image_size': args.image_size,
            'metrics': best_metrics or {},
        },
        best_checkpoint_path,
    )
    metrics_path.write_text(
        json.dumps(
            {
                'manifest': str(args.manifest),
                'image_root': str(args.image_root),
                'train_rows': len(train_rows),
                'validation_rows': len(validation_rows),
                'device': str(device),
                'augment': args.augment,
                'trim_ink': args.trim_ink,
                'weighted_loss': args.weighted_loss,
                'history': history,
                'best': best_metrics or {},
                'final': history[-1] if history else {},
            },
            indent=2,
        )
        + '\n',
        encoding='utf-8',
    )
    args_path.write_text(
        json.dumps(
            {
                'manifest': str(args.manifest),
                'image_root': str(args.image_root),
                'out_dir': str(out_dir),
                'run_name': out_dir.name,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'image_size': args.image_size,
                'validation_fraction': args.validation_fraction,
                'seed': args.seed,
                'augment': args.augment,
                'trim_ink': args.trim_ink,
                'weighted_loss': args.weighted_loss,
                'requested_device': args.device,
                'resolved_device': str(device),
            },
            indent=2,
        )
        + '\n',
        encoding='utf-8',
    )

    print(f'wrote checkpoint: {checkpoint_path}')
    print(f'wrote best checkpoint: {best_checkpoint_path}')
    print(f'wrote metrics: {metrics_path}')
    print(f'wrote args: {args_path}')
    return 0


def _is_better(metrics: dict[str, float], best_metrics: dict[str, float]) -> bool:
    return (
        metrics['exact_accuracy'],
        metrics['rank_accuracy'],
        metrics['suit_accuracy'],
    ) > (
        best_metrics['exact_accuracy'],
        best_metrics['rank_accuracy'],
        best_metrics['suit_accuracy'],
    )


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.out_dir is not None:
        return args.out_dir
    run_name = args.run_name or _default_run_name(args)
    return DEFAULT_RUNS_ROOT / run_name


def _default_run_name(args: argparse.Namespace) -> str:
    parts = [
        args.device,
        f'lr{_format_float_for_name(args.learning_rate)}',
        f'seed{args.seed}',
    ]
    if args.augment:
        parts.append('aug')
    if args.trim_ink:
        parts.append('trim')
    if args.weighted_loss:
        parts.append('weighted')
    return '-'.join(parts)


def _format_float_for_name(value: float) -> str:
    return f'{value:g}'.replace('.', 'p').replace('-', 'm')


def _select_device(device: str) -> torch.device:
    if device == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    requested = torch.device(device)
    if requested.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but is not available')
    if requested.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS requested but is not available')
    return requested


if __name__ == '__main__':
    raise SystemExit(main())
