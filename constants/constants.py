import os
import torch

TWITTER_RAW_DATA = os.path.join(os.getcwd(), "data", "twitter", "raw", "twitter_dataset.csv")
TWITTER_PREPROCESSED_DATA = os.path.join(
    os.getcwd(), "data", "twitter", "preprocessed", "twitter_preprocessed_data.csv"
)

GEOTEXT_RAW_DATA = os.path.join(os.getcwd(), "data", "geotext", "raw", "GeoTextDataset.csv")
GEOTEXT_PREPROCESSED_DATA = os.path.join(
    os.getcwd(), "data", "geotext", "preprocessed", "GeoTextPreprocessedDataset.csv"
)

TWITTER_EDGE_PATH = os.path.join(os.getcwd(), "data", "twitter", "edges")

TWITTER_GRAPH_PATH = os.path.join(os.getcwd(), "data", "twitter", "graph_data")
TWITTER_LM_DATA_PATH = os.path.join(os.getcwd(), "data", "twitter", "lm_data")
TWITTER_GRAPH_MODEL_PATH = os.path.join(os.getcwd(), "models", "twitter", "graph_models")
TWITTER_LANGUAGE_MODEL_PATH = os.path.join(os.getcwd(), "models", "twitter", "lm_models")


device = "cuda" if torch.cuda.is_available() else "cpu"

TWITTER_LM_TRAINING_RESULT_PATH = os.path.join(
    os.getcwd(), "results", "lm_training_results"
)

seed = 42  # Random seed for reproducibility
