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
        tasks: List,
        optimize: str = "memory",
        validation_texts: List[str] = None,
        validation_split: float = 0.2,
        **training_args,
    ):

        config = AutoConfig.from_pretrained(self.base_encoder)

        encoder = AutoModel.from_pretrained(self.base_encoder, config=config)

        # initialize tasks and collect labels

        #  TODO handling training args is still WIP
        training_args = self.prepare_training_args(training_args)

        # train_loader = DataLoader(
        #     dataset, batch_size=training_args["batch_size"], shuffle=True
        # )

        self.model = MultiHeadModel(
            encoder=encoder,
            tasks=tasks,
            learning_rate=training_args["learning_rate"],
        )

        # TODO must improve next 3 lines
        gpus = 1 if self.device.type.startswith("cuda") else 0
        training_args.pop("learning_rate")
        training_args.pop("batch_size")

        trainer = pl.Trainer(
            gpus=gpus, logger=False, checkpoint_callback=False, **training_args
        )
        trainer.fit(self.model)

        return self.model

    def predict(self, texts, optimize: str = "memory", **testing_args):
        raise NotImplementedError()

    def score(self, texts):
        raise NotImplementedError()

    def cross_validate(self, texts):
        raise NotImplementedError()

    def prepare_training_args(self, training_args):
        defaults = {"learning_rate": 2e-5, "max_epochs": 20, "batch_size": 4}
        defaults.update(training_args)
        return defaults


class MultiHeadModel(pl.LightningModule):
    """
    Build a composite model made of a base encoder and several classification heads.
    """

    def __init__(self, encoder, tasks, learning_rate) -> None:
        super().__init__()
        self.encoder = encoder

        self.heads = nn.ModuleDict({t.name: t.head for t in tasks})
        self.tasks = {task.name: task for task in tasks}

        self.save_hyperparameters("learning_rate")

    def train_dataloader(self):
        loaders = []

        for t in self.tasks.values():
            loaders.append(torch.utils.data.DataLoader(t.dataset, batch_size=4))

        return loaders

    def forward(self, input_ids: dict, **encoder_kwargs):
        out = self.encoder(input_ids, **encoder_kwargs)[1]
        return out

    def training_step(self, batch, batch_idx):

        loss = []
        for inner_batch in batch:
            input_ids = inner_batch["input_ids"]
            attention_mask = inner_batch["attention_mask"]
            labels = inner_batch["labels"]

            hidden = self(input_ids, attention_mask=attention_mask)

            output = self.heads[inner_batch["name"][0]](hidden)

            loss.append(self.tasks[inner_batch["name"][0]].loss(labels, output))

        return torch.stack(loss).sum()

    def configure_optimizers(self):
        optim = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optim

labels_A = [0, 1, 0, 1, 0, 1, 0, 1]
labels_B = [1, 0, 1, 0, 1, 0, 1, 0]

task_A = ClassificationTask("bert-base-uncased", texts=["test", "gigi"] * 4, labels=labels_A, optimize="compute", name="task1")
task_B = ClassificationTask("bert-base-uncased", texts=["test", "gigi"] * 4, labels=labels_B, optimize="compute", name="task2")

t = Tuner("bert-base-uncased", "bert-base-uncased")
t.fit(tasks=[task_A, task_B], optimize="compute", batch_size=4)

