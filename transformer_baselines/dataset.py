from torch.utils.data import Dataset
import torch


class SingleTaskDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class TokenizingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, **tokenizer_kwargs):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs

    def __getitem__(self, idx):
        item = self.tokenizer(self.texts[idx], **self.tokenizer_kwargs)
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
