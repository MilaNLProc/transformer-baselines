from transformer_baselines.heads.DistilBert import DistilBertHead
from .Bert import BertHead
from .DistilBert import DistilBertHead


class AutoHead:
    @staticmethod
    def from_pretrained(model_name, config, task_type):

        if model_name.startswith("bert"):
            return BertHead(config, task_type)

        if model_name.startswith("distilbert"):
            return DistilBertHead(config, task_type)

        else:
            raise NotImplementedError()
