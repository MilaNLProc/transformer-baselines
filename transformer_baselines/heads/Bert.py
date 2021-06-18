import torch.nn as nn


class BertHead(nn.Module):
    def __init__(self, config, task_type):
        super(BertHead, self).__init__()
        self.task_type = task_type
        self.config = config

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, encoder_outputs):

        if self.task_type == "sentence_classification":
            x = encoder_outputs["pooler_output"]
            x = self.dropout(x)
            logits = self.classifier(x)
            return logits

        elif self.task_type == "ner":
            x = encoder_outputs["last_hidden_state"]
            x = self.dropout(x)
            logits = self.classifier(x)
            return logits

        else:
            raise NotImplementedError()