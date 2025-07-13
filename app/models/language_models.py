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
from app.core.graph_related.graph import get_feature_embeddings


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
    if tokenizer is None:
        return get_feature_embeddings(texts)
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
        #implement dropout
        # self.dropout = nn.Dropout(p=0.3)

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

#Basic Multi Layer Perceptron (MLP) model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, labels=None):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            labels = labels.float().unsqueeze(1)
            loss = loss_fn(x, labels)

        return {"logits": x, "loss": loss}


bert_config = BertConfig()
bert_model = CustomBertModel( BertModel(bert_config)).to(device)

distilbert_config = DistilBertConfig()
distilbert_model = CustomBertModel(DistilBertModel(distilbert_config)).to(device)

roberta_config = RobertaConfig()
roberta_model = CustomBertModel(RobertaModel(roberta_config)).to(device)

# Load pre-trained models
pre_trained_bert = BertModel.from_pretrained('bert-base-uncased')
pre_trained_bert_model = CustomBertModel(pre_trained_bert).to(device)

pre_trained_distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
pre_trained_distilbert_model = CustomBertModel(pre_trained_distilbert).to(device)

pre_trained_roberta = RobertaModel.from_pretrained('roberta-base')
pre_trained_roberta_model = CustomBertModel(pre_trained_roberta).to(device)


