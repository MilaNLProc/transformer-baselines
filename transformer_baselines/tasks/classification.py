from transformers import AutoModelForSequenceClassification
from transformers.models.auto.configuration_auto import AutoConfig


class ClassificationTask:
    def __init__(self, model_name, labels) -> None:
        self.labels = labels
        self.model_name = model_name

    def instantiate(self):
        config = AutoConfig.from_pretrained(
            self.model_name, num_labels=len(set(self.labels)), finetuning_task="custom"
        )

        self.head = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, config=config
        ).classifier
