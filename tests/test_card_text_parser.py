import pytest

from ai_balatro.utils.card_text_parser import parse_card_description


@pytest.mark.parametrize(
    'raw_text, expected_rank, expected_suit, expected_short',
    [
        ('方片K\n+10筹码', 'K', 'diamonds', 'KD'),
        ('黑桃7\n+7筹码', '7', 'spades', '7S'),
        ('海化2\n+2筹码', '2', 'clubs', '2C'),
        ('红挑A', 'A', 'hearts', 'AH'),
    ],
)
def test_parse_card_description_honors_rank_and_suit(
    raw_text: str, expected_rank: str, expected_suit: str, expected_short: str
) -> None:
    result = parse_card_description(raw_text)
    assert result is not None
    assert result['valid'] is True
    assert result['rank'] == expected_rank
    assert result['suit'] == expected_suit
    assert result['short_code'] == expected_short
    assert expected_suit.title() in result['english_name']


def test_parse_card_description_gracefully_handles_empty_text() -> None:
    assert parse_card_description('') is None
    assert parse_card_description('   ') is None
