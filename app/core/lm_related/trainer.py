import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from app.scripts.train_eval_lm import (
    get_training_arguments,
    trainer,
    train_model,
    evaluate,
)
from app.models.language_models import (
    bert_model,
    distilbert_model,
    roberta_model,
)
from constants import TWITTER_LM_DATA_PATH, TWITTER_LANGUAGE_MODEL_PATH


def train(
    data_type, model_name: str = "bert", learning_rate: float = 1e-5, epochs: int = 10
):
    """
    Trains a language model with the specified parameters.

    Args:
        model (str): The model to be trained (default is "bert").
        learning_rate (float): Learning rate for training (default is 1e-5).
        epochs (int): Number of epochs for training (default is 10).
    """
    try:
        train_dataset = torch.load(
            os.path.join(TWITTER_LM_DATA_PATH, f"{data_type}_train_dataset_{model_name}.pt")
        )
        val_dataset = torch.load(
            os.path.join(TWITTER_LM_DATA_PATH, f"{data_type}_val_dataset_{model_name}.pt")
        )
        test_dataset = torch.load(
            os.path.join(TWITTER_LM_DATA_PATH, f"{data_type}_test_dataset_{model_name}.pt")
        )

        # training_args = get_training_arguments(model, learning_rate, epochs)

        model = (
            distilbert_model
            if model_name == "distillbert"
            else (roberta_model if model_name == "roberta" else bert_model)
        )

        # Dataloaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)
        test_loader = DataLoader(test_dataset, batch_size=16)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=5e-4
        )
        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}")
            train_model(model, train_loader, optimizer)
            val_acc, val_recall, val_f1, val_precision = evaluate(model, val_loader)
            # Print validation metrics in pretty format
            print(
                f"Validation - Accuracy: {val_acc:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, Precision: {val_precision:.4f}"
            )
            # Save the model after each epoch
        model_save_path = os.path.join(TWITTER_LANGUAGE_MODEL_PATH, f"{data_type}_{model_name}.pt")
        print(f"Saving model to {model_save_path}...")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
        test_acc, test_recall, test_f1, test_precision = evaluate(model, test_loader)
        # Print test metrics in pretty format
        print(
            f"Test - Accuracy: {test_acc:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}, Precision: {test_precision:.4f}"
        )
        # trainer_instance = trainer(
        #     model, train_dataset, val_dataset, training_args
        # )
        # trainer_instance.train()
        print(
            f"Training completed for {model_name} with learning rate {learning_rate} for {epochs} epochs."
        )
    except Exception as e:
        print(f"Error during training: {e}")
