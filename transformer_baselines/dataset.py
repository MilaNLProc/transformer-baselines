from torch.utils.data import Dataset
import torch
from typing import List


class OptimizedTaskDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = [t[idx] for t in self.labels]
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


class TokenizingDataset(Dataset):
    def __init__(self, texts: List[str], labels: List, tokenizer, **tokenizer_kwargs):
        self.texts = texts
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs

        # this is List[List[int]]
        self.labels = labels

    def __getitem__(self, idx):

        item = self.tokenizer(self.texts[idx], **self.tokenizer_kwargs)

        item["labels"] = torch.tensor([t[idx] for t in self.labels])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
