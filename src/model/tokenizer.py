from tokenizers import ByteLevelBPETokenizer
import os

class GPT2Tokenizer:
    def __init__(self, vocab_path=None, merges_path=None):
        if vocab_path and merges_path:
            # Load existing tokenizer
            self.tokenizer = ByteLevelBPETokenizer(vocab_path, merges_path)
        else:
            # Create empty tokenizer for training
            self.tokenizer = ByteLevelBPETokenizer()

    def train(self, texts, vocab_size=50257, save_dir="tokenizers/gpt_tokenizer"):
        # Train from an iterator of text samples
        self.tokenizer.train_from_iterator(texts, vocab_size=vocab_size)

        # Ensure destination folder exists
        os.makedirs(save_dir, exist_ok=True)

        # Correct save call (no tokenizer argument!)
        self.tokenizer.save_model(save_dir)

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)
