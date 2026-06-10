"""Small CNN classifier for Balatro card-corner rank and suit labels."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
SUITS = ['spades', 'hearts', 'clubs', 'diamonds']
RANK_TO_INDEX = {rank: index for index, rank in enumerate(RANKS)}
SUIT_TO_INDEX = {suit: index for index, suit in enumerate(SUITS)}


@dataclass(frozen=True)
class LabeledCardCorner:
    """One readable card-corner sample."""

    image: str
    rank: str
    suit: str


def load_labeled_rows(manifest_path: Path) -> list[LabeledCardCorner]:
    """Load readable rank/suit labels from a JSONL manifest."""

    rows = []
    for line in manifest_path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue

        item = json.loads(line)
        rank = item.get('rank')
        suit = item.get('suit')
        if item.get('quality') != 'readable' or rank not in RANK_TO_INDEX:
            continue
        if suit not in SUIT_TO_INDEX:
            continue

        rows.append(LabeledCardCorner(image=item['image'], rank=rank, suit=suit))

    return rows


class CardCornerDataset(Dataset):
    """PyTorch dataset for labeled Balatro card-corner crops."""

    def __init__(
        self,
        *,
        rows: list[LabeledCardCorner],
        image_root: Path,
        image_size: int = 64,
        augment: bool = False,
        trim_ink: bool = False,
    ) -> None:
        self.rows = rows
        self.image_root = image_root
        self.image_size = image_size
        self.augment = augment
        self.trim_ink = trim_ink

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image = Image.open(self.image_root / row.image).convert('RGB')
        if self.trim_ink:
            image = trim_to_ink(image)
        if self.augment:
            image = _augment_image(image)
        image = image.resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        tensor = torch.tensor(list(image.tobytes()), dtype=torch.float32)
        tensor = tensor.reshape(self.image_size, self.image_size, 3).permute(2, 0, 1)
        tensor = tensor / 255.0

        return (
            tensor,
            torch.tensor(RANK_TO_INDEX[row.rank], dtype=torch.long),
            torch.tensor(SUIT_TO_INDEX[row.suit], dtype=torch.long),
        )


class CardCornerClassifier(nn.Module):
    """Tiny shared-backbone CNN with separate rank and suit heads."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.rank_head = nn.Linear(256, len(RANKS))
        self.suit_head = nn.Linear(256, len(SUITS))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.classifier(self.backbone(images))
        return self.rank_head(features), self.suit_head(features)


def split_rows(
    rows: list[LabeledCardCorner],
    *,
    validation_fraction: float = 0.2,
    seed: int = 7,
) -> tuple[list[LabeledCardCorner], list[LabeledCardCorner]]:
    """Deterministically split rows into train and validation partitions."""

    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    validation_size = max(1, round(len(shuffled) * validation_fraction))
    return shuffled[validation_size:], shuffled[:validation_size]


def class_weights(
    rows: list[LabeledCardCorner],
    *,
    label: str,
) -> torch.Tensor:
    """Return inverse-frequency class weights for rank or suit labels."""

    if label == 'rank':
        values = RANKS
        index_by_value = RANK_TO_INDEX
        counts = Counter(row.rank for row in rows)
    elif label == 'suit':
        values = SUITS
        index_by_value = SUIT_TO_INDEX
        counts = Counter(row.suit for row in rows)
    else:
        raise ValueError("label must be 'rank' or 'suit'")

    weights = torch.zeros(len(values), dtype=torch.float32)
    nonzero_counts = [count for count in counts.values() if count > 0]
    if not nonzero_counts:
        return weights

    mean_count = sum(nonzero_counts) / len(nonzero_counts)
    for value in values:
        count = counts[value]
        if count > 0:
            weights[index_by_value[value]] = mean_count / count
    return weights


def trim_to_ink(image: Image.Image, *, padding: int = 4) -> Image.Image:
    """Crop an image around dark or saturated rank/suit pixels."""

    rgb = image.convert('RGB')
    data = rgb.tobytes()
    xs = []
    ys = []
    width, height = rgb.size
    for y in range(height):
        for x in range(width):
            index = (y * width + x) * 3
            red = data[index]
            green = data[index + 1]
            blue = data[index + 2]
            dark = red <= 90 and green <= 90 and blue <= 90
            saturated = max(red, green, blue) - min(red, green, blue) >= 70
            bright_enough = max(red, green, blue) >= 120
            if dark or (saturated and bright_enough):
                xs.append(x)
                ys.append(y)

    if not xs:
        return rgb

    left = max(0, min(xs) - padding)
    top = max(0, min(ys) - padding)
    right = min(width, max(xs) + padding + 1)
    bottom = min(height, max(ys) + padding + 1)
    return rgb.crop((left, top, right, bottom))


@torch.no_grad()
def evaluate(
    model: CardCornerClassifier,
    loader,
    *,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate rank, suit, and exact-card accuracy."""

    model.eval()
    total = 0
    rank_correct = 0
    suit_correct = 0
    exact_correct = 0
    for images, ranks, suits in loader:
        images = images.to(device)
        ranks = ranks.to(device)
        suits = suits.to(device)
        rank_logits, suit_logits = model(images)
        rank_pred = rank_logits.argmax(dim=1)
        suit_pred = suit_logits.argmax(dim=1)
        total += ranks.numel()
        rank_correct += (rank_pred == ranks).sum().item()
        suit_correct += (suit_pred == suits).sum().item()
        exact_correct += ((rank_pred == ranks) & (suit_pred == suits)).sum().item()

    if total == 0:
        return {'rank_accuracy': 0.0, 'suit_accuracy': 0.0, 'exact_accuracy': 0.0}

    return {
        'rank_accuracy': rank_correct / total,
        'suit_accuracy': suit_correct / total,
        'exact_accuracy': exact_correct / total,
    }


def _augment_image(image: Image.Image) -> Image.Image:
    image = image.rotate(
        random.uniform(-4.0, 4.0),
        resample=Image.Resampling.BILINEAR,
        expand=False,
    )
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
    return ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.20))
