from torch.utils.data import Dataset
import torch
from typing import List
import datasets


class OptimizedTaskDataset(Dataset):
    def __init__(self, encodings, name, labels=None):
        self.encodings = encodings
        self.labels = labels
        self.name = name

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        item["name"] = self.name
        return item

    def __len__(self):
        return len(self.labels)


class TokenizingDataset(Dataset):
    def __init__(self, texts: List[str], labels: List, tokenizer, name, **tokenizer_kwargs):
        self.texts = texts
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs
        self.labels = labels
        self.name = name

    def __getitem__(self, idx):
        item = self.tokenizer(self.texts[idx], **self.tokenizer_kwargs)
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        item["name"] = self.name
        return item

    def __len__(self):
        return len(self.labels)


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


def build_dataset(texts, tokenizer, task_labels, optimize, name):
    if optimize == "memory":
        dataset = build_optimized_memory_dataset(
            texts,
            tokenizer,
            task_labels,
            padding="max_length",  #  TODO we can optimize here
            truncation=True,
            return_tensors="pt",
        )

    elif optimize == "compute":
        encodings = tokenizer(
            texts, truncation=True, padding=True, return_tensors="pt"
        )
        dataset = OptimizedTaskDataset(encodings, labels=task_labels, name=name)
    else:
        raise ValueError(f"'optimize' value {optimize} is not supported.")

    return dataset

