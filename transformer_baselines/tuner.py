from typing import List, Dict
from transformers import AutoConfig, AutoTokenizer, AutoModel, AdamW
import torch.nn as nn
from torch.utils.data import DataLoader
from transformer_baselines.tasks import ClassificationTask
from transformer_baselines.dataset import (
    TokenizingDataset,
    OptimizedTaskDataset,
    build_optimized_memory_dataset,
)
import torch
from tqdm import tqdm
import logging


logging.basicConfig(
    format="%(levelname)s:%(asctime)s:%(module)s:%(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class Tuner:
    def __init__(self, base_encoder: str, base_tokenizer: str, device="auto") -> None:
        self.base_encoder = base_encoder
        self.base_tokenizer = base_tokenizer

        if device == "auto":
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = device

        logging.info(f"Using device: {self.device}")

    def fit(
        self,
        texts: List[str],
        tasks: List,
        optimize: str = "memory",
        validation_texts: List[str] = None,
        validation_split: float = 0.2,
        **training_args,
    ):

        config = AutoConfig.from_pretrained(self.base_encoder)
        tokenizer = AutoTokenizer.from_pretrained(self.base_tokenizer)
        encoder = AutoModel.from_pretrained(self.base_encoder, config=config)

        # initialize tasks and collect labels
        tasks_labels = list()
        for t in tasks:
            t.initialize(self.device)
            tasks_labels.append(t.labels)

        if optimize == "memory":
            dataset = build_optimized_memory_dataset(
                texts,
                tokenizer,
                tasks_labels,
                padding="max_length",  #  TODO we can optimize here
                truncation=True,
                return_tensors="pt",
            )

        elif optimize == "compute":
            encodings = tokenizer(
                texts, truncation=True, padding=True, return_tensors="pt"
            )

            dataset = OptimizedTaskDataset(encodings, tasks_labels)
        else:
            raise ValueError(f"'optimize' value {optimize} is not supported.")

        self.model = MultiHeadModel(encoder, [t.head for t in tasks])

        self.model.to(self.device)
        self.model.train()

        training_args = TrainingArgs(training_args)
        train_loader = DataLoader(
            dataset, batch_size=training_args["batch_size"], shuffle=True
        )

        optim = AdamW(self.model.parameters(), lr=training_args["learning_rate"])

        pbar = tqdm(
            total=training_args["max_epochs"], position=0, leave=True, desc="Epochs"
        )

        for epoch in range(training_args["max_epochs"]):

            pbar.update(1)
            for batch in train_loader:

                optim.zero_grad()

                input_ids = batch["input_ids"].to(self.device)

                attention_mask = batch["attention_mask"].to(self.device)
                for i in range(0, len(batch["labels"])):
                    batch["labels"][i] = batch["labels"][i].to(self.device)

                labels = batch["labels"]

                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = torch.tensor(0.0, device=self.device)

                for output, label, t in zip(outputs, labels, tasks):
                    loss += t.loss(label, output)

                loss.backward()
                optim.step()

        pbar.close()

        return self.model

    def predict(self, texts, optimize: str = "memory", **training_args):
        tokenizer = AutoTokenizer.from_pretrained(self.base_tokenizer)

        if optimize == "memory":
            dataset = build_optimized_memory_dataset(
                texts,
                tokenizer,
                padding="max_length",  #  TODO we can optimize here
                truncation=True,
                return_tensors="pt",
            )
        elif optimize == "compute":
            encodings = tokenizer(
                texts, truncation=True, padding=True, return_tensors="pt"
            )

            dataset = OptimizedTaskDataset(encodings)
        else:
            raise NotImplementedError()

        training_args = TrainingArgs(training_args)
        train_loader = DataLoader(dataset, batch_size=training_args["batch_size"])
        self.model.eval()
        collect_outputs = [[], []]  # TODO: MALE
        for batch in train_loader:

            input_ids = batch["input_ids"].to(self.device)

            attention_mask = batch["attention_mask"].to(self.device)

            outputs = self.model(input_ids, attention_mask=attention_mask)

            for i in range(0, len(outputs)):
                collect_outputs[i].extend(
                    torch.argmax(outputs[i], axis=1).detach().cpu().numpy().tolist()
                )

        return collect_outputs

    def score(self, texts):
        raise NotImplementedError()

    def cross_validate(self, texts):
        raise NotImplementedError()


class MultiHeadModel(nn.Module):
    """
    Build a composite model made of a base encoder and several classification heads.
    """

    def __init__(self, encoder, heads) -> None:
        super().__init__()
        self.encoder = encoder
        self.heads = heads

    def forward(self, input_ids: dict, **encoder_kwargs):
        out = self.encoder(input_ids, **encoder_kwargs)[1]
        return [h(out) for h in self.heads]


class TrainingArgs:
    DEFAULT_ARGS = {"learning_rate": 2e-5, "max_epochs": 20, "batch_size": 4}

    def __init__(self, training_args: Dict) -> None:
        self.args = TrainingArgs.DEFAULT_ARGS
        for k, v in training_args.items():
            self.args[k] = v

    def __getitem__(self, name):
        return self.args[name]
