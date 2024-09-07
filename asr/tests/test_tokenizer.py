import pytest

from whisper.tokenizer import get_tokenizer


@pytest.mark.parametrize("multilingual", [True, False])
def test_tokenizer(multilingual):
    tokenizer = get_tokenizer(multilingual=False)
    assert tokenizer.sot in tokenizer.sot_sequence


def test_multilingual_tokenizer():
    gpt2_tokenizer = get_tokenizer(multilingual=False)
    multilingual_tokenizer = get_tokenizer(multilingual=True)

    text = "다람쥐 헌 쳇바퀴에 타고파"
    gpt2_tokens = gpt2_tokenizer.encode(text)
    multilingual_tokens = multilingual_tokenizer.encode(text)

    assert gpt2_tokenizer.decode(gpt2_tokens) == text
    assert multilingual_tokenizer.decode(multilingual_tokens) == text
    assert len(gpt2_tokens) > len(multilingual_tokens)
