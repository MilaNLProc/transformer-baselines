from transformers import AutoModelForTokenClassification
from transformers.models.auto.configuration_auto import AutoConfig
from transformers import AutoTokenizer
from torch.nn import CrossEntropyLoss
from transformer_baselines.dataset import *
import numpy as np

class NERTask:
    def __init__(self, texts, labels, optimize="compute"):

        self.texts = texts
        # TODO we might think of using sklearn's LabelEncoder here for broader support
        self.labels = labels
        self.optimize = optimize
        self.loss_function = CrossEntropyLoss()
        self.hidden_idx = 0

        self.unique_tags = set(tag for doc in labels for tag in doc)
        self.num_labels = len(set(self.unique_tags))

        self.tag2id = {tag: id for id, tag in enumerate(self.unique_tags)}
        self.id2tag = {id: tag for tag, id in self.tag2id.items()}

    def initialize(self, model_name, tokenizer):
        config = AutoConfig.from_pretrained(
           model_name, num_labels=self.num_labels, finetuning_task="custom"
        )

        self.head = AutoModelForTokenClassification.from_pretrained(
            model_name, config=config
        ).classifier

        self.dataset = build_ner_dataset(self.texts, tokenizer, self.labels, self.optimize, self.tag2id)

    def loss(self, labels, logits, attention_mask):
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels))

                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return loss



