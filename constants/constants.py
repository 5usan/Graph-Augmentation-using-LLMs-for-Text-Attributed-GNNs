import os
import torch

TWITTER_RAW_DATA = os.path.join(os.getcwd(), "data", "raw", "twitter_dataset.csv")
TWITTER_PREPROCESSED_DATA = os.path.join(
    os.getcwd(), "data", "preprocessed", "twitter_preprocessed_data.csv"
)

GEOTEXT_RAW_DATA = os.path.join(os.getcwd(), "data", "raw", "GeoTextDataset.csv")
GEOTEXT_PREPROCESSED_DATA = os.path.join(
    os.getcwd(), "data", "preprocessed", "GeoTextPreprocessedDataset.csv"
)

EDGE_PATH = os.path.join(os.getcwd(), "data", "edges")

GRAPH_PATH = os.path.join(os.getcwd(), "data", "graph_data")
GRAPH_MODEL_PATH = os.path.join(os.getcwd(), "models", "graph_models")
LANGUAGE_MODEL_PATH = os.path.join(os.getcwd(), "models", "lm_models")

LM_DATA_PATH = os.path.join(os.getcwd(), "data", "lm_data")

device = "cuda" if torch.cuda.is_available() else "cpu"

LM_TRAINING_RESULT_PATH = os.path.join(
    os.getcwd(), "results", "lm_training_results"
)

seed = 42  # Random seed for reproducibility
