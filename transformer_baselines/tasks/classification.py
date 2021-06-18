from transformers import AutoModelForSequenceClassification
from transformers.models.auto.configuration_auto import AutoConfig
from transformers import AutoTokenizer
from torch.nn import CrossEntropyLoss
from transformer_baselines.dataset import *

# TODO: Task should be an abstract class
class ClassificationTask:
    def __init__(self, texts, labels, optimize="compute"):

        self.texts = texts
        self.hidden_idx = 1
        # TODO we might think of using sklearn's LabelEncoder here for broader support
        self.labels = labels
        self.optimize = optimize
        self.num_labels = len(set(labels))
        self.loss_function = CrossEntropyLoss()

    def initialize(self, model_name, tokenizer):
        config = AutoConfig.from_pretrained(
           model_name, num_labels=self.num_labels, finetuning_task="custom"
        )

        self.head = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config
        ).classifier

        self.dataset = build_dataset(self.texts, tokenizer, self.labels, self.optimize)

    def loss(self, labels, logits, attention_mask=None):
        loss = self.loss_function(logits.view(-1, self.num_labels), labels.view(-1))
        return loss
