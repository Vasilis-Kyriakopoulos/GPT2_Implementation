# tests/test_tokenizer.py
from src.model.tokenizer import GPT2Tokenizer

def test_encode_decode_roundtrip():
    # assuming trained tokenizer loaded here
    tok = GPT2Tokenizer()
    text = "The war started"
    ids = tok.encode(text)
    out = tok.decode(ids)
    assert isinstance(ids, list)
    assert isinstance(out, str)

