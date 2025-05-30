import os
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from app.models.language_models import bert_tokenizer, distilbert_tokenizer, roberta_tokenizer, tokenizer_function
from app.dataset.twitter_dataset import TwitterDataset

from constants import (
    GRAPH_PATH,
    LM_DATA_PATH,
)


def label_encode_data(dataset: pd.DataFrame):
    """
    Encodes the labels in the dataset using LabelEncoder.

    Args:
        dataset (pd.DataFrame): The dataset containing features and labels.

    Returns:
        pd.DataFrame: The dataset with encoded labels.
    """
    label_encoder = LabelEncoder()
    dataset["label"] = label_encoder.fit_transform(dataset["label"].tolist())
    return dataset


def split_data(data_type: str):
    """
    Splits data into training and testing sets based on the specified data type.

    Args:
        data_type (str): Type of data to split (e.g., "twitter", "geotext").

    Returns:
        dict: A dictionary containing the split data or an error message.
    """
    try:
        if data_type not in ["twitter", "geotext"]:
            return {"error": "Invalid data type. Choose either 'twitter' or 'geotext'."}
        train_data_path = os.path.join(GRAPH_PATH, f"graph_{data_type}_train_data.csv")
        val_data_path = os.path.join(GRAPH_PATH, f"graph_{data_type}_val_data.csv")
        test_data_path = os.path.join(GRAPH_PATH, f"graph_{data_type}_test_data.csv")
        train_dataset = pd.read_csv(train_data_path)
        val_dataset = pd.read_csv(val_data_path)
        test_dataset = pd.read_csv(test_data_path)
        print(f"Loaded train data from {train_data_path}")
        print(f"Loaded validation data from {val_data_path}")
        print(f"Loaded test data from {test_data_path}")

        train_label_encoded = label_encode_data(train_dataset)
        val_label_encoded = label_encode_data(val_dataset)
        test_label_encoded = label_encode_data(test_dataset)
        print("Labels encoded successfully.")

        train_path = os.path.join(LM_DATA_PATH, f"{data_type}_train_data.csv")
        val_path = os.path.join(LM_DATA_PATH, f"{data_type}_val_data.csv")
        test_path = os.path.join(LM_DATA_PATH, f"{data_type}_test_data.csv")

        train_label_encoded.to_csv(train_path)
        val_label_encoded.to_csv(val_path)
        test_label_encoded.to_csv(test_path)
        print(f"Data saved to {train_path}, {val_path}, and {test_path}")

        return {"message": f"Data for {data_type} split successfully."}

    except Exception as e:
        print(e)
        print(f"Error splitting data for {data_type}: {e}")
        return {"error": str(e)}


def create_data(data_type: str, model: str = "bert"):
    """
    Creates a dataset by splitting the data into training, validation, and test sets.

    Args:
        data_type (str): Type of data to create (e.g., "twitter", "geotext").

    """
    try:
        if data_type not in ["twitter", "geotext"]:
            return {"error": "Invalid data type. Choose either 'twitter' or 'geotext'."}

        train_data_path = os.path.join(LM_DATA_PATH, f"{data_type}_train_data.csv")
        val_data_path = os.path.join(LM_DATA_PATH, f"{data_type}_val_data.csv")
        test_data_path = os.path.join(LM_DATA_PATH, f"{data_type}_test_data.csv")

        train_data = pd.read_csv(train_data_path)
        val_data = pd.read_csv(val_data_path)
        test_data = pd.read_csv(test_data_path)
        print(f"Loaded data for {data_type}")
        tokenizer = distilbert_tokenizer if model == "distillbert" else (
            roberta_tokenizer if model == "roberta" else tokenizer
        )
        train_encodings = tokenizer_function(
            train_data["feature"].tolist(), tokenizer
        )
        val_encodings = tokenizer_function(val_data["feature"].tolist(), tokenizer)
        test_encodings = tokenizer_function(
            test_data["feature"].tolist(), tokenizer
        )
        print("Tokenization completed.")

        train_dataset = TwitterDataset(train_encodings, train_data["label"].tolist())
        val_dataset = TwitterDataset(val_encodings, val_data["label"].tolist())
        test_dataset = TwitterDataset(test_encodings, test_data["label"].tolist())

        # Saving the datasets
        train_dataset_path = os.path.join(LM_DATA_PATH, f"{data_type}_train_dataset_{model}.pt")
        val_dataset_path = os.path.join(LM_DATA_PATH, f"{data_type}_val_dataset_{model}.pt")
        test_dataset_path = os.path.join(LM_DATA_PATH, f"{data_type}_test_dataset_{model}.pt")
        torch.save(train_dataset, train_dataset_path)
        torch.save(val_dataset, val_dataset_path)
        torch.save(test_dataset, test_dataset_path)
        print("Datasets created successfully.")
        return {"message": f"Data for {data_type} created successfully."}

    except Exception as e:
        print(f"Error creating data for {data_type}: {e}")
        return {"error": str(e)}
