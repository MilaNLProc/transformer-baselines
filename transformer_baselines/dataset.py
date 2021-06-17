from torch.utils.data import Dataset
import torch
from typing import List
import datasets


class OptimizedTaskDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return self.encodings["input_ids"].shape[0]


# class TokenizingDataset(Dataset):
#     def __init__(
#         self, texts: List[str], labels: List, tokenizer, name, **tokenizer_kwargs
#     ):
#         self.texts = texts
#         self.tokenizer = tokenizer
#         self.tokenizer_kwargs = tokenizer_kwargs
#         self.labels = labels

#     def __getitem__(self, idx):
#         item = self.tokenizer(self.texts[idx], **self.tokenizer_kwargs)
#         if self.labels:
#             item["labels"] = torch.tensor(self.labels[idx])
#         return item

#     def __len__(self):
#         return len(self.labels)


def build_optimized_memory_dataset(texts, tokenizer, labels=None, **tokenizer_kwargs):
    data = dict()

    data["texts"] = texts
    data["labels"] = labels

    dataset = datasets.Dataset.from_dict(data)

    def encode(batch):
        item = tokenizer(batch["texts"], **tokenizer_kwargs)
        if labels:
            item["labels"] = batch["labels"]

        return item

    dataset.set_transform(encode)
    return dataset


def build_dataset(texts, tokenizer, task_labels, optimize):
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
        encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
        dataset = OptimizedTaskDataset(encodings, labels=task_labels)
    else:
        raise ValueError(f"'optimize' value {optimize} is not supported.")

    return dataset
