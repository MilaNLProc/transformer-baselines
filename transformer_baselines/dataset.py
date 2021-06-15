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
        self.labels = labels

    def __getitem__(self, idx):
        item = self.tokenizer(self.texts[idx], **self.tokenizer_kwargs)
        item["labels"] = torch.tensor([t[idx] for t in self.labels])
        return item

    def __len__(self):
        return len(self.texts)


def build_optimized_memory_dataset(
    texts, tokenizer, tasks_labels=None, **tokenizer_kwargs
):
    data = dict()

    data["texts"] = texts
    if tasks_labels:
        for tid, labels in enumerate(tasks_labels):
            data[f"labels_{tid}"] = labels

    dataset = datasets.Dataset.from_dict(data)

    def encode(batch):
        item = tokenizer(batch["texts"], **tokenizer_kwargs)

        if tasks_labels:
            item["labels"] = [
                batch[f"labels_{tid}"] for tid in range(len(tasks_labels))
            ]

        return item

    dataset.set_transform(encode)
    return dataset
