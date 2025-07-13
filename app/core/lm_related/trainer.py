import os
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader


from app.scripts.train_eval_lm import (
    get_training_arguments,
    trainer,
    train_model,
    evaluate,
    train_mlp_model,
    evaluate_mlp,
)

from constants.constants import (
    device,
)
from app.models.language_models import (
    bert_model,
    distilbert_model,
    roberta_model,
    pre_trained_bert_model,
    pre_trained_distilbert_model,
    pre_trained_roberta_model,
    MLP,
)
from constants import TWITTER_LM_DATA_PATH, TWITTER_LANGUAGE_MODEL_PATH


def train(
    data_type,
    model_name: str = "bert",
    learning_rate: float = 1e-5,
    epochs: int = 10,
    preTrained: bool = False,
):
    """
    Trains a language model with the specified parameters.

    Args:
        data_type (str): Type of data to train on (e.g., "twitter", "geotext").
        model_name (str): The name of the model to be trained (default is "bert").
        learning_rate (float): Learning rate for training (default is 1e-5).
        epochs (int): Number of epochs for training (default is 10).
        preTrained (bool): Whether to use a pre-trained model (default is False).
    """
    try:
        train_dataset = torch.load(
            os.path.join(
                TWITTER_LM_DATA_PATH, f"{data_type}_train_dataset_{model_name}.pt"
            )
        )
        val_dataset = torch.load(
            os.path.join(
                TWITTER_LM_DATA_PATH, f"{data_type}_val_dataset_{model_name}.pt"
            )
        )
        test_dataset = torch.load(
            os.path.join(
                TWITTER_LM_DATA_PATH, f"{data_type}_test_dataset_{model_name}.pt"
            )
        )

        print(train_dataset, "train_dataset-----------")

        train_data_path = os.path.join(
            TWITTER_LM_DATA_PATH, f"{data_type}_train_data.csv"
        )
        train_data = pd.read_csv(train_data_path)
        # training_args = get_training_arguments(model, learning_rate, epochs)

        # Initialize the model based on the model name
        if model_name == "bert":
            model = bert_model
            pre_trained_model = pre_trained_bert_model
        elif model_name == "distillbert":
            model = distilbert_model
            pre_trained_model = pre_trained_distilbert_model
        elif model_name == "roberta":
            model = roberta_model
            pre_trained_model = pre_trained_roberta_model
        elif model_name == "MLP":
            input_dim = 384
            hidden_dim = 128
            output_dim = 1
            model = MLP(input_dim, hidden_dim, output_dim).to(device)
        else:
            raise ValueError(
                "Invalid model name. Choose from 'bert', 'distillbert', 'mlp', or 'roberta'."
            )
        # If preTrained is True, use the pre-trained model
        if preTrained:
            model = pre_trained_model

        # Dataloaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)
        test_loader = DataLoader(test_dataset, batch_size=16)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=0.001
        )
        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}")
            if model_name == "MLP":
                train_mlp_model(model, train_loader, optimizer)
                val_acc, val_recall, val_f1, val_precision = evaluate_mlp(
                    model, val_loader
                )
            else:
                train_model(model, train_loader, optimizer)
                val_acc, val_recall, val_f1, val_precision = evaluate(model, val_loader)
            # Print validation metrics in pretty format
            print(
                f"Validation - Accuracy: {val_acc:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, Precision: {val_precision:.4f}"
            )
            # Save the model after each epoch
        model_save_path = os.path.join(
            TWITTER_LANGUAGE_MODEL_PATH, f"{data_type}_{model_name}.pt"
        )
        print(f"Saving model to {model_save_path}...")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
        test_acc, test_recall, test_f1, test_precision = (
            evaluate(model, test_loader)
            if model_name != "MLP"
            else evaluate_mlp(model, test_loader)
        )
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
