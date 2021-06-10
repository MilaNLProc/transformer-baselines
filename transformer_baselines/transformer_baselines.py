"""Main module."""
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch
from transformer_baselines.dataset import SingleTaskDataset
from transformers import AdamW
from torch.utils.data import DataLoader
import os
from tqdm import tqdm


class SingleTaskTuner:
    def __init__(self, model_name, tokenizer_name):
        """
        Defines a single task model
        """
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def cross_validate_fit(self, texts, labels, splits=5, epochs=5, batch_size=16, learning_rate=5e-5):
        """
        Uses Sklearn cross validation module to run a stratified cross-validation pipeline
        """

        config = AutoConfig.from_pretrained(self.model_name,
                                            num_labels=len(set(labels)),
                                            finetuning_task="custom")
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        texts = np.array(texts)
        labels = np.array(labels)

        skf = StratifiedKFold(n_splits=splits)

        original = []
        predicted = []

        for train_index, test_index in skf.split(texts, labels):
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=config)

            X_train, X_test = texts[train_index].tolist(), texts[test_index].tolist()
            y_train, y_test = labels[train_index].tolist(), labels[test_index].tolist()

            #TODO not the smartest way to do this, but faster to code up
            #TODO there is probably the need to replace this with the lazy version
            tokenized_train = tokenizer(X_train, truncation=True, padding=True)
            tokenized_test = tokenizer(X_test, truncation=True, padding=True)

            train_dataset = SingleTaskDataset(tokenized_train, y_train)
            test_dataset = SingleTaskDataset(tokenized_test, y_test)

            model.to(self.device)
            model.train()

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            optim = AdamW(model.parameters(), lr=learning_rate)

            pbar = tqdm(total=epochs, position=0, leave=True)
            for epoch in range(epochs):
                pbar.update(1)
                for batch in train_loader:
                    optim.zero_grad()
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    lab = batch['labels'].to(self.device)
                    outputs = model(input_ids, attention_mask=attention_mask, labels=lab)

                    loss = outputs[0]
                    loss.backward()
                    optim.step()
            pbar.close()
            model.eval()

            loader = DataLoader(test_dataset, batch_size=batch_size)
            original.extend(y_test)
            with torch.no_grad():
                for batch in loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)

                    outputs = model(input_ids, attention_mask=attention_mask)
                    predicted.extend(torch.argmax(outputs["logits"], axis=1).cpu().numpy().tolist())
            del model
        return original, predicted

    def train_and_save(self, texts, labels, path, epochs=5, batch_size=16, learning_rate=5e-5):
        """
        Trains the model and saves it on disk
        """

        config = AutoConfig.from_pretrained(self.model_name, num_labels=len(set(labels)),
                                            finetuning_task="custom")

        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=config)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        tokenized_train = tokenizer(texts, truncation=True, padding=True)

        train_dataset = SingleTaskDataset(tokenized_train, labels)

        model.to(self.device)
        model.train()

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optim = AdamW(model.parameters(), lr=learning_rate)

        pbar = tqdm(total=epochs, position=0, leave=True)
        for epoch in range(epochs):
            pbar.update(1)
            for batch in train_loader:
                optim.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                lab = batch['labels'].to(self.device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=lab)
                loss = outputs[0]
                loss.backward()
                optim.step()
        pbar.close()

        os.makedirs(path)
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
