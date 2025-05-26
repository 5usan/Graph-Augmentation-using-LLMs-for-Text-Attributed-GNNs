from transformers import (
    BertModel,
    BertTokenizer,
    Trainer,
    TrainingArguments,
    DistilBertModel,
    DistilBertTokenizer,
    RobertaModel,
    RobertaTokenizer,
    EarlyStoppingCallback,
)


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
        max_length = 128,
        return_tensors='pt'  
    )