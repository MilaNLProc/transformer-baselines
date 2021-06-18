from torch.utils.data import Dataset
import torch
import datasets
import numpy as np

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
        return len(self.labels)



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


def build_dataset(texts, tokenizer, task_labels, optimize, return_offsets_mapping=False):
    if optimize == "memory":
        dataset = build_optimized_memory_dataset(
            texts,
            tokenizer,
            task_labels,
            padding="max_length",  #  TODO we can optimize here
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=return_offsets_mapping
        )

    elif optimize == "compute":
        encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt",
                              return_offsets_mapping=return_offsets_mapping)
        dataset = OptimizedTaskDataset(encodings, labels=task_labels)
    else:
        raise ValueError(f"'optimize' value {optimize} is not supported.")

    return dataset


def encode_tags(tags, encodings, tag2id):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

def build_ner_dataset(texts, tokenizer, task_labels, optimize, tag2id):
    if optimize == "memory":
        raise NotImplementedError()
        # dataset = build_optimized_memory_dataset(
        #     texts,
        #     tokenizer,
        #     task_labels,
        #     padding="max_length",  #  TODO we can optimize here
        #     truncation=True,
        #     return_tensors="pt",
        #     return_offsets_mapping=True
        # )
    elif optimize == "compute":
        encodings = tokenizer(texts, is_split_into_words=True, truncation=True, padding=True, return_tensors="pt",
                              return_offsets_mapping=True)

        encoded_tags = encode_tags(task_labels, encodings, tag2id)
        encodings.pop("offset_mapping")
        dataset = OptimizedTaskDataset(encodings, labels=encoded_tags)
    else:
        raise ValueError(f"'optimize' value {optimize} is not supported.")

    return dataset
