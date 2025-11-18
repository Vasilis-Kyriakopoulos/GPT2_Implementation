import torch
from torch.utils.data import Dataset


class GPT2Dataset(Dataset):
    def __init__(self,text, tokenizer,block_size=128):
        ids = tokenizer.encode(text)
        self.data = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

