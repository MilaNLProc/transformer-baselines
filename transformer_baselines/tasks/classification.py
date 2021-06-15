from transformers import AutoModelForSequenceClassification
from transformers.models.auto.configuration_auto import AutoConfig
from transformers import AutoTokenizer
from torch.nn import CrossEntropyLoss
from transformer_baselines.dataset import *


class ClassificationTask:
    def __init__(self, model_name, texts, labels, optimize, name) -> None:

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.texts = texts
        self.name = name
        self.num_labels = len(set(labels))
        self.loss_function = CrossEntropyLoss()
        self.dataset = build_dataset(texts, self.tokenizer, labels, optimize, name)
        self.initialize()

        # TODO we might think of using sklearn's LabelEncoder here for broader support
        self.labels = labels

    def initialize(self):

        config = AutoConfig.from_pretrained(
            self.model_name, num_labels=self.num_labels, finetuning_task="custom"
        )

        self.head = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, config=config
        ).classifier

    def loss(self, labels, logits):
        loss = self.loss_function(logits.view(-1, self.num_labels), labels.view(-1))
        return loss
