import os
import torch
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

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

from app.utils.utility import compute_metrics
from constants import TWITTER_LM_TRAINING_RESULT_PATH, device


def get_training_arguments(model="bert", learning_rate=1e-5, epochs=10):
    """
    Returns the training arguments for the Trainer.

    Returns:
        TrainingArguments: The training arguments configured for the Trainer.
    """

    training_args = TrainingArguments(
        output_dir=os.path.join(TWITTER_LM_TRAINING_RESULT_PATH, f"./{model}_Results"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        # warmup_ratio=0.06,  # Prevent aggressive weight updates early
        # gradient_accumulation_steps=2,  # Simulate larger batch without increasing memory
        # fp16=True,  # Use mixed precision training if available
    )

    return training_args


def trainer(model, train_dataset, val_dataset, training_args):
    """
    Initializes and returns a Trainer instance for model training.

    Args:
        model: The model to be trained.
        train_dataset: The dataset for training.
        val_dataset: The dataset for validation.
        training_args: The training arguments.

    Returns:
        Trainer: An instance of the Trainer class.
    """
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    return trainer


def train_model(model, train_loader, optimizer):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device).float()

        output = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = output["loss"]
        loss.backward()
        optimizer.step()

        optimizer.step()


def evaluate(model, loader):
    model.eval()
    preds, true = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).float().unsqueeze(1)

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(output["logits"])
            pred = (probs > 0.5).long()

            preds.extend(pred.cpu().numpy())
            true.extend(labels.cpu().numpy())

    print("Sample probs:", probs[:5].squeeze().cpu().numpy())
    print("Sample pred: ", pred[:5].squeeze().cpu().numpy())
    print("Sample true: ", labels[:5].squeeze().cpu().numpy())
    # return accracy, recall, f1, precision

    return accuracy_score(true, preds), recall_score(true, preds), f1_score(true, preds), precision_score(true, preds)
