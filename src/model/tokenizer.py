from tokenizers import ByteLevelBPETokenizer


class GPT2Tokenizer:
    def __init__(self, vocab_path=None, merges_path=None):
        if vocab_path:
            self.tokenizer = ByteLevelBPETokenizer(vocab_path, merges_path)
        else:
            self.tokenizer = ByteLevelBPETokenizer()

    def train(self, texts, vocab_size=50257):
        self.tokenizer.train_from_iterator(texts, vocab_size=vocab_size)
#        self.tokenizer.save_model("./tokenizer")

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)
