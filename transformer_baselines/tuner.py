from typing import List
from transformers import AutoConfig, AutoTokenizer, AutoModel, AdamW
import torch.nn as nn
from torch.utils.data import DataLoader
from .tasks import ClassificationTask
from .dataset import TokenizingDataset


class Tuner:
    def __init__(self, base_encoder: str, base_tokenizer: str) -> None:
        self.base_encoder = base_encoder
        self.base_tokenizer = base_tokenizer
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def fit(
        self,
        texts: List[str],
        tasks: List,
        optimize: str = "memory",
        val_size=0.2,
        **training_args
    ):

        config = AutoConfig.from_pretrained(self.base_encoder)
        tokenizer = AutoTokenizer.from_pretrained(self.base_tokenizer)
        encoder = AutoModel.from_pretrained(self.base_encoder, config=config)

        # initialize tasks and collect labels
        labels = list()
        for t in tasks:
            task.initialize()
            labels.append(t.labels)

        # our datasets will use a Tuple as "labels" in the batch
        labels = zip(labels)

        if optimize == "memory":
            # tokenized texts on the fly
            dataset = TokenizingDataset(
                texts, labels, tokenizer, trucation=True, padding=True
            )
        elif optimize == "compute":
            # tokenize everything upfront
            raise NotImplementedError()
        else:
            raise NotImplementedError()

        model = DummyModel(encoder, [t.head for t in tasks])

        model.to(self.device)
        model.train()

        # TODO define a training - validation split using "val_size" % of the data

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optim = AdamW(model.parameters(), lr=learning_rate)

        pbar = tqdm(total=epochs, position=0, leave=True)
        for epoch in range(epochs):
            pbar.update(1)
            for batch in train_loader:
                optim.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = model(input_ids, attention_mask=attention_mask)

                # TODO manually compute a loss per-head output & sum them
                # loss = ...

                loss.backward()
                optim.step()
        pbar.close()

    def predict(self, texts):
        raise NotImplementedError()

    def score(self, texts):
        raise NotImplementedError()

    def cross_validate(self, texts):
        raise NotImplementedError()


class DummyModel(nn.Module):
    """
    Build a composite model made of a base encoder and several classification heads.
    """

    def __init__(self, encoder, heads) -> None:
        self.encoder = encoder
        self.heads = heads

    def forward(self, input_ids: dict, **encoder_kwargs):
        out = self.encoder(input_ids, **encoder_kwargs)

        # TODO produce per-head output
        raise NotImplementedError()
