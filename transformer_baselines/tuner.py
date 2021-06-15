from typing import List, Dict
from transformers import AutoConfig, AutoTokenizer, AutoModel, AdamW
import torch.nn as nn
from torch.utils.data import DataLoader
from transformer_baselines.tasks import ClassificationTask
from transformer_baselines.dataset import (
    OptimizedTaskDataset,
    build_optimized_memory_dataset,
)
import torch
from tqdm import tqdm
import logging
import pytorch_lightning as pl


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

        #  TODO handling training args is still WIP
        training_args = self.prepare_training_args(training_args)

        train_loader = DataLoader(
            dataset, batch_size=training_args["batch_size"], shuffle=True
        )

        self.model = MultiHeadModel(
            encoder=encoder,
            tasks=tasks,
            learning_rate=training_args["learning_rate"],
        )

        # TODO must improve next 3 lines
        gpus = 1 if self.device.type.startswith("cuda") else 0
        training_args.pop("learning_rate")
        training_args.pop("batch_size")

        trainer = pl.Trainer(gpus=gpus, **training_args)
        trainer.fit(self.model, train_loader)

        return self.model

    def predict(self, texts, optimize: str = "memory", **testing_args):
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

        testing_args = self.prepare_training_args(testing_args)
        test_loader = DataLoader(dataset, batch_size=testing_args["batch_size"])

        with torch.no_grad():
            self.model.to(self.device)
            self.model.eval()

            collect_outputs = [[], []]  # TODO: MALE
            for batch in test_loader:

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

    def prepare_training_args(self, training_args):
        defaults = {"learning_rate": 2e-5, "max_epochs": 10, "batch_size": 4}
        defaults.update(training_args)
        return defaults


class MultiHeadModel(pl.LightningModule):
    """
    Build a composite model made of a base encoder and several classification heads.
    """

    def __init__(self, encoder, tasks, learning_rate) -> None:
        super().__init__()
        self.encoder = encoder
        self.tasks = tasks

        self.save_hyperparameters("learning_rate")

    def forward(self, input_ids: dict, **encoder_kwargs):
        out = self.encoder(input_ids, **encoder_kwargs)[1]
        return [t.head(out) for t in self.tasks]

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self(input_ids, attention_mask=attention_mask)

        tasks_losses = [
            t.loss(label, output)
            for output, label, t in zip(outputs, labels, self.tasks)
        ]

        loss = torch.stack(tasks_losses).sum()

        return loss

    def configure_optimizers(self):
        params = [{"params": self.encoder.parameters()}]
        for t in self.tasks:
            params.append({"params": t.head.parameters()})

        optim = AdamW(params, lr=self.hparams.learning_rate)
        return optim
