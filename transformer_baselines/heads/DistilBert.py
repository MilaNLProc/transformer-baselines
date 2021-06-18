import torch.nn as nn


class DistilBertHead(nn.Module):
    def __init__(self, config, task_type):
        super(DistilBertHead, self).__init__()
        self.task_type = task_type
        self.config = config

        # for sequence classification
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        #  for NER
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, encoder_outputs):

        if self.task_type == "sentence_classification":
            hidden_state = encoder_outputs["last_hidden_state"]  # (bs, seq_len, dim)
            pooled_output = hidden_state[:, 0]  # (bs, dim)
            pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
            pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
            pooled_output = self.dropout(pooled_output)  # (bs, dim)
            logits = self.classifier(pooled_output)  # (bs, num_labels)
            return logits

        elif self.task_type == "ner":
            x = encoder_outputs["last_hidden_state"]
            x = self.dropout(x)
            logits = self.classifier(x)
            return logits

        else:
            raise NotImplementedError()