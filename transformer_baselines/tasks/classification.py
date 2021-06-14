from transformers import AutoModelForSequenceClassification
from transformers.models.auto.configuration_auto import AutoConfig


class ClassificationTask:
    def __init__(self, model_name, targets) -> None:
        self.targets = targets
        self.model_name = model_name

    def instantiate(self):
        config = AutoConfig.from_pretrained(
            self.model_name, num_labels=len(set(self.targets)), finetuning_task="custom"
        )

        self.head = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, config=config
        ).classifier
