import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from app.utils.utility import create_csv, get_data
from app.core.prompt_related.llm_api import predict_label
from constants.constants import TWITTER_EDGE_PATH


def build_edges(data, feature_embeddings, threshold=0.5):
    """
    Build edges based on cosine similarity's threshold only.

    Args:
        data (dict): Dictionary containing the features and labels.
        feature_embeddings (torch.Tensor): Feature embeddings for the data.
        threshold (float): Threshold for cosine similarity.

    Returns:
        list: List of edges based on cosine similarity.
    """
    try:
        print("Building edges based on cosine similarity...")
        similarity_matrix = cosine_similarity(feature_embeddings.to("cpu"))
        edges = []
        for i in range(len(data["feature"])):
            # print(f"Processing feature {i}/{len(data["feature"])}")
            for j in range(i, len(data["feature"])):
                if i == j:
                    edges.append((i, j))
                else:
                    if similarity_matrix[i][j] > threshold:
                        edges.append((i, j))
                        edges.append((j, i))
        # Save edges to CSV
        edges_df = pd.DataFrame(edges, columns=["source", "target"])
        create_csv(
            edges_df,
            os.path.join(TWITTER_EDGE_PATH, f"edges_cosine_similarity_{threshold}.csv"),
        )
        print("Edges built successfully.")
        return edges
    except Exception as e:
        print(f"An error occurred while generating edges: {e}")
        return []


def build_edges_by_predicting_labels(
    data, label, feature_embeddings, threshold=0.5, model="deepseek-r1:8b"
):
    """
    Build edges based on llm api predictions.

    Args:
        data (dict): Dictionary containing the features and labels.
        label (str): Label to consider for classification.
        feature_embeddings (torch.Tensor): Feature embeddings for the data.
        threshold (float): Threshold for cosine similarity.
        model (str): Model name for the LLM API.

    Returns:
        list: List of edges based on llm api predictions.
    """
    predicted_values = []
    PREDICTED_GENDER_VALUES_PATH = os.path.join(
        TWITTER_EDGE_PATH, f"predicted_{label}_values.csv"
    )

    PREDICTED_EDGE_PATH_WITH_THRESHOLD = os.path.join(
        TWITTER_EDGE_PATH, f"edges_{label}_prediction_{threshold}.csv"
    )
    PREDICTED_EDGE_PATH = os.path.join(TWITTER_EDGE_PATH, f"edges_{label}_prediction.csv")
    try:
        print(f"model: {model}")
        similarity_matrix = cosine_similarity(feature_embeddings.to("cpu"))
        edges = []
        if os.path.isfile(PREDICTED_GENDER_VALUES_PATH):
            data_df = get_data(PREDICTED_GENDER_VALUES_PATH)
            print(data_df.head(50))
            if os.path.isfile(PREDICTED_EDGE_PATH):
                print(f"Edges already exist at {PREDICTED_EDGE_PATH}")
                edges_df = pd.read_csv(PREDICTED_EDGE_PATH)
                for i in range(len(edges_df)):
                    if edges_df["source"][i] != edges_df["target"][i]:
                        edges.append((edges_df["source"][i], edges_df["target"][i]))
                return edges

            for i in range(len(data_df)):
                print(f"Processing feature {i} of {len(data_df)}")
                if float(data_df["predicted_confidence"][i]) >= 0.8:
                    for j in range(i, len(data_df)):
                        if similarity_matrix[i][j] > threshold:
                            if i == j:
                                edges.append((i, j))
                            elif (
                                data_df["predicted_label"][i]
                                == data_df["predicted_label"][j]
                                and float(data_df["predicted_confidence"][j]) >= 0.8
                            ):
                                edges.append((i, j))
                                edges.append((j, i))
        else:
            for i in range(len(data["feature"])):
                print(f"Processing feature {i} of {len(data['feature'])}")
                ith_predicted_result = predict_label(data, i, model, label)
                predicted_values.append(
                    [
                        data["feature"][i],
                        data["label"][i],
                        (
                            ith_predicted_result["label"]
                            if ith_predicted_result["label"]
                            else ""
                        ),
                        (
                            ith_predicted_result["confidence"]
                            if ith_predicted_result["confidence"]
                            else ""
                        ),
                        (
                            ith_predicted_result["explaination"]
                            if ith_predicted_result["explaination"]
                            else ""
                        ),
                    ],
                )
            predicted_values_df = pd.DataFrame(
                predicted_values,
                columns=[
                    "feature",
                    "label",
                    "predicted_label",
                    "predicted_confidence",
                    "explaination",
                ],
            )
            create_csv(
                predicted_values_df,
                PREDICTED_GENDER_VALUES_PATH,
            )
        print("Edges built successfully.")
        # Save edges to CSV
        edges_df = pd.DataFrame(edges, columns=["source", "target"])
        create_csv(
            edges_df,
            os.path.join(TWITTER_EDGE_PATH, f"edges_{label}_prediction_{threshold}.csv"),
        )
        return edges
    except Exception as e:
        print(f"An error occurred while building edges: {e}")
        return []
