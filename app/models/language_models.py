import torch.nn as nn
from transformers import (
    BertModel,
    BertTokenizer,
    DistilBertModel,
    DistilBertTokenizer,
    RobertaModel,
    RobertaTokenizer,
    BertConfig,
    DistilBertConfig,
    RobertaConfig,
)

from constants import device


bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


def tokenizer_function(texts, tokenizer):
    """
    Tokenizes a list of texts using the specified tokenizer.
    Args:
        texts (list of str): List of texts to tokenize.
        tokenizer: The tokenizer to use (BertTokenizer, DistilBertTokenizer, or RobertaTokenizer).
    Returns:
        dict: A dictionary containing the tokenized inputs, suitable for model input.
    """
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )


class CustomBertModel(nn.Module):
    def __init__(self, pre_trained_model):
        super().__init__()
        self.bert = pre_trained_model

        # Enable training on BERT layers
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            labels = labels.float().unsqueeze(1)
            loss = self.loss_fn(logits, labels)

        return {"logits": logits, "loss": loss}

bert_config = BertConfig()
bert_model = CustomBertModel( BertModel(bert_config)).to(device)

distilbert_config = DistilBertConfig()
distilbert_model = CustomBertModel(DistilBertModel(distilbert_config)).to(device)

roberta_config = RobertaConfig()
roberta_model = CustomBertModel(RobertaModel(roberta_config)).to(device)
