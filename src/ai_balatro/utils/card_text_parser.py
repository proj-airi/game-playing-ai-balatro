"""Utilities for normalizing and parsing Balatro card description text."""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Optional

# Canonical suit metadata used for normalization
SUITS: Dict[str, Dict[str, str]] = {
    'diamonds': {'name': 'Diamonds', 'symbol': '♦', 'code': 'D'},
    'hearts': {'name': 'Hearts', 'symbol': '♥', 'code': 'H'},
    'spades': {'name': 'Spades', 'symbol': '♠', 'code': 'S'},
    'clubs': {'name': 'Clubs', 'symbol': '♣', 'code': 'C'},
}

RANK_NAMES: Dict[str, str] = {
    'A': 'Ace',
    'K': 'King',
    'Q': 'Queen',
    'J': 'Jack',
    '10': 'Ten',
    '9': 'Nine',
    '8': 'Eight',
    '7': 'Seven',
    '6': 'Six',
    '5': 'Five',
    '4': 'Four',
    '3': 'Three',
    '2': 'Two',
}

# Common RapidOCR mis-recognitions mapped to their intended tokens
_SEQUENCE_SUBSTITUTIONS = {
    '红挑': '红桃',
    '黑挑': '黑桃',
    '黑优': '黑桃',
    '海化': '梅花',
    '梅华': '梅花',
    '梅化': '梅花',
    '方申': '方片',
    '芳片': '方片',
}

_CHAR_SUBSTITUTIONS = {
    '：': ':',
    '；': ';',
    '，': ',',
    '。': '.',
    '（': '(',
    '）': ')',
}

_SUIT_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), suit)
    for pattern, suit in [
        (r'(方片|方块|DIAMOND|DIAMONDS|♦)', 'diamonds'),
        (r'(红桃|红心|HEART|HEARTS|♥)', 'hearts'),
        (r'(黑桃|SPADE|SPADES|♠)', 'spades'),
        (r'(梅花|CLUB|CLUBS|♣)', 'clubs'),
    ]
]

_RANK_PATTERN = re.compile(r'(10|[2-9]|[AKQJ]|T)', re.IGNORECASE)


def _normalize_text(text: str) -> str:
    """Return a trimmed, NFKC-normalized version of the input text."""
    normalized = unicodedata.normalize('NFKC', text)

    for src, dst in _SEQUENCE_SUBSTITUTIONS.items():
        normalized = normalized.replace(src, dst)

    for src, dst in _CHAR_SUBSTITUTIONS.items():
        normalized = normalized.replace(src, dst)

    return normalized.strip()


def _extract_primary_line(text: str) -> str:
    """Take the most informative line for rank/suit parsing."""
    if not text:
        return ''

    # Many tooltips encode the card on the first line before score modifiers
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else text.strip()


def _detect_suit(text: str) -> Optional[str]:
    """Detect suit keyword from normalized text."""
    for pattern, suit in _SUIT_PATTERNS:
        if pattern.search(text):
            return suit
    return None


def _extract_rank(text: str) -> Optional[str]:
    """Extract rank token from normalized text."""
    match = _RANK_PATTERN.search(text)
    if not match:
        return None

    token = match.group(1).upper()
    if token == 'T':
        return '10'
    return token


def parse_card_description(description_text: str) -> Optional[Dict[str, Optional[str]]]:
    """Parse OCR tooltip text into structured card metadata.

    Args:
        description_text: Raw OCR text captured from the tooltip

    Returns:
        Mapping with normalized metadata (rank, suit, etc.) or None if unusable.
    """
    if not description_text:
        return None

    normalized_text = _normalize_text(description_text)
    if not normalized_text:
        return None

    primary_text = _extract_primary_line(normalized_text)

    suit = _detect_suit(primary_text)
    rank = _extract_rank(primary_text)

    result: Dict[str, Optional[str]] = {
        'raw_text': description_text,
        'normalized_text': normalized_text,
        'primary_text': primary_text,
        'suit': suit,
        'rank': rank,
        'suit_name': None,
        'suit_symbol': None,
        'short_code': None,
        'english_name': None,
        'valid': None,
    }

    if suit and suit in SUITS:
        suit_meta = SUITS[suit]
        result['suit_name'] = suit_meta['name']
        result['suit_symbol'] = suit_meta['symbol']

    if rank and rank in RANK_NAMES:
        if suit:
            card_name = f"{RANK_NAMES[rank]} of {SUITS[suit]['name']}"
            result['english_name'] = card_name
            result['short_code'] = f"{rank}{SUITS[suit]['code']}"
        else:
            result['english_name'] = RANK_NAMES[rank]

    result['valid'] = bool(result['rank'] and result['suit'])

    return result


__all__ = ['parse_card_description', 'SUITS', 'RANK_NAMES']
