from types import prepare_class
from typing import List, Dict
from transformers import AutoConfig, AutoTokenizer, AutoModel, AdamW
import torch.nn as nn
from torch.utils.data import DataLoader
from transformer_baselines.tasks import ClassificationTask
from transformer_baselines.dataset import (
    OptimizedTaskDataset,
    build_dataset,
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

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_tokenizer)

    def fit(
        self,
        tasks: List,
        **training_args,
    ):

        config = AutoConfig.from_pretrained(self.base_encoder)

        encoder = AutoModel.from_pretrained(self.base_encoder, config=config)

        #  TODO handling training args is still WIP
        training_args = include_defaults(training_args, "train")

        for t in tasks:
            t.initialize(self.base_encoder, self.tokenizer)

        self.model = MultiHeadModel(
            encoder=encoder,
            tasks=tasks,
            learning_rate=training_args["learning_rate"],
            batch_size=training_args["batch_size"],
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

    def predict(self, texts, optimize: str = "compute", **testing_args):
        testing_args = include_defaults(testing_args, "test")

        dataset = build_dataset(texts, self.tokenizer, None, optimize)
        test_dataloader = DataLoader(dataset, batch_size=testing_args["batch_size"])

        preds = self._run_test(test_dataloader)
        return preds

    def score(self, texts):
        raise NotImplementedError()

    def cross_validate(self, texts):
        raise NotImplementedError()

    def _run_test(self, dataloader):
        with torch.no_grad():

            self.model.to(self.device)
            self.model.eval()

            batch_outputs = list()
            for batch in dataloader:

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                hidden = self.model(input_ids, attention_mask=attention_mask)

                #  N_tasks x B x 2
                outputs = torch.stack([head(hidden) for head in self.model.heads])
                batch_outputs.append(outputs)

            preds = torch.cat(batch_outputs, 1).argmax(-1).detach().cpu().tolist()
            return preds


def include_defaults(args, split):
    if split == "train":
        defaults = {"learning_rate": 2e-5, "max_epochs": 20, "batch_size": 4}
    elif split == "test":
        defaults = {"batch_size": 4}
    else:
        raise ValueError(f"{split} is not a valid split")

    defaults.update(args)
    return defaults


class MultiHeadModel(pl.LightningModule):
    """
    Build a composite model made of a base encoder and several classification heads.
    """

    def __init__(self, encoder, tasks, learning_rate, batch_size) -> None:
        super().__init__()
        self.encoder = encoder

        self.heads = nn.ModuleList([t.head for t in tasks])
        self.tasks = tasks

        self.save_hyperparameters("learning_rate", "batch_size")

    def train_dataloader(self):
        loaders = [
            DataLoader(t.dataset, batch_size=self.hparams.batch_size)
            for t in self.tasks
        ]
        return loaders

    def forward(self, input_ids: dict, **encoder_kwargs):
        out = self.encoder(input_ids, **encoder_kwargs)[1]
        return out

    def training_step(self, batch, batch_idx):

        loss = []
        for tid, inner_batch in enumerate(batch):
            input_ids = inner_batch["input_ids"]
            attention_mask = inner_batch["attention_mask"]
            labels = inner_batch["labels"]

            hidden = self(input_ids, attention_mask=attention_mask)

            output = self.heads[tid](hidden)

            loss.append(self.tasks[tid].loss(labels, output))

        return torch.stack(loss).sum()

    def configure_optimizers(self):
        optim = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optim
