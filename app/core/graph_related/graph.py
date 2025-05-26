import os
import torch
import random
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from torch_geometric.transforms import RandomNodeSplit, RandomLinkSplit

from constants.constants import (
    GEOTEXT_PREPROCESSED_DATA,
    TWITTER_PREPROCESSED_DATA,
    GRAPH_PATH,
    device,
    seed,
)
from app.utils.utility import get_data
from app.core.graph_related.create_edges import (
    build_edges,
    build_edges_by_predicting_labels,
)


def get_sentence_bert_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load a Sentence-BERT model from the Hugging Face Model Hub.

    Args:
        model_name (str): Name of the model to load.

    Returns:
        SentenceTransformer: Loaded Sentence-BERT model.
    """
    try:
        model = SentenceTransformer(model_name)
        print(f"Model {model_name} loaded successfully.")
        return model
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None


def get_feature_embeddings(sentences):
    """
    Get feature embeddings for a list of sentences using the provided model.

    Args:
        model (SentenceTransformer): Loaded Sentence-BERT model.
        sentences (list): List of sentences to get embeddings for.

    Returns:
        list: List of feature embeddings.
    """
    try:
        model = get_sentence_bert_model()
        embeddings = model.encode(
            sentences, show_progress_bar=True, convert_to_tensor=True, device="cuda"
        )  # Change device to generic function
        print("Feature embeddings generated successfully.")
        return embeddings
    except Exception as e:
        print(f"An error occurred while generating embeddings: {e}")
        return None


def generate_graph(data_type: str):
    """
    Create a graph based on the specified data type.

    Args:
        data_type (str): Type of data to create graph (e.g., "twitter", "geotext").
    """
    try:
        preprocessed_data = pd.DataFrame()
        # Load the preprocessed dataset with the feature and label columns
        if data_type == "twitter":
            preprocessed_data = get_data(TWITTER_PREPROCESSED_DATA)
        elif data_type == "geotext":
            preprocessed_data = get_data(GEOTEXT_PREPROCESSED_DATA)
        if preprocessed_data.empty:
            return {"error": "No data to create graph."}

        # Extract feature embeddings
        feature_embeddings = get_feature_embeddings(
            preprocessed_data["feature"].tolist()
        )

        # Get label tensor
        label_encoder = LabelEncoder()
        label_encoded = label_encoder.fit_transform(preprocessed_data["label"].tolist())
        label_tensor = torch.tensor(label_encoded, dtype=torch.long).unsqueeze(1)
        for index, label in enumerate(label_encoder.classes_):
            print(f"{index}: {label}")
        print("Label tensor created.")
        node_features = feature_embeddings
        print(f"Node features is of size: {node_features.size()}")
        node_labels = label_tensor

        # Build edges for the graph
        # edges = build_edges(
        #     preprocessed_data,
        #     feature_embeddings,
        #     threshold=0.40,
        # )
        edges = build_edges_by_predicting_labels(
            preprocessed_data,
            "gender",
            feature_embeddings,
            threshold=0.50,
            model="gemma3:latest",
        )

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Create graph
        graph = Data(x=node_features, edge_index=edge_index, y=node_labels)
        print("Graph created successfully.")
        torch.save(graph, os.path.join(GRAPH_PATH, f"graph_{data_type}.pt"))
        return graph
    except Exception as e:
        return {"error": str(e)}


def split_graph(data_type: str, train_ratio: float = 0.8):
    """
    Split the graph into training, validation and testing sets.

    Args:
        data_type (str): type of graph (twitter, geotext).
        train_ratio (float): The ratio of the training set size to the total size.

    Returns:
        True if the graph is split successfully, False otherwise.
    """
    try:
        # Load the graph
        graph_data = torch.load(
            os.path.join(GRAPH_PATH, f"graph_{data_type}.pt"),
            map_location=torch.device(device),
        )
        print(f"Graph loaded successfully from {GRAPH_PATH}.")
        print(f"Graph data: {graph_data}")
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Split the graph into train, validation and test sets
        splitter = RandomNodeSplit(split="train_rest", num_val=0.2, num_test=0.2)
        splitted_graph_data = splitter(graph_data)
        print(f"Graph data after splitting: {splitted_graph_data}")

        torch.save(
            splitted_graph_data,
            os.path.join(GRAPH_PATH, f"graph_{data_type}_splitted.pt"),
        )

        # Get text from the feature embeddings of training, validation and test set and save in csv file with labels
        if data_type == "twitter":
            preprocessed_data = get_data(TWITTER_PREPROCESSED_DATA)
        elif data_type == "geotext":
            preprocessed_data = get_data(GEOTEXT_PREPROCESSED_DATA)

        train_indices = splitted_graph_data.train_mask.nonzero(as_tuple=True)[0]
        val_indices = splitted_graph_data.val_mask.nonzero(as_tuple=True)[0]
        test_indices = splitted_graph_data.test_mask.nonzero(as_tuple=True)[0]

        train_texts = preprocessed_data.iloc[train_indices]["feature"].tolist()
        train_labels = preprocessed_data.iloc[train_indices]["label"].tolist()

        val_texts = preprocessed_data.iloc[val_indices]["feature"].tolist()
        val_labels = preprocessed_data.iloc[val_indices]["label"].tolist()

        test_texts = preprocessed_data.iloc[test_indices]["feature"].tolist()
        test_labels = preprocessed_data.iloc[test_indices]["label"].tolist()

        # Save the train, validation and test data in csv files
        train_df = pd.DataFrame(
            {"indices": train_indices, "feature": train_texts, "label": train_labels}
        )
        train_df.to_csv(
            os.path.join(GRAPH_PATH, f"graph_{data_type}_train_data.csv"), index=False
        )

        val_df = pd.DataFrame(
            {"indices": val_indices, "feature": val_texts, "label": val_labels}
        )
        val_df.to_csv(
            os.path.join(GRAPH_PATH, f"graph_{data_type}_val_data.csv"), index=False
        )

        test_df = pd.DataFrame(
            {"indices": test_indices, "feature": test_texts, "label": test_labels}
        )
        test_df.to_csv(
            os.path.join(GRAPH_PATH, f"graph_{data_type}_test_data.csv"), index=False
        )

        return True
    except Exception as e:
        print(f"An error occurred while splitting graph: {e}")
        return False
