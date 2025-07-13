import os
import ast
import tqdm
import torch
import random
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from torch_geometric.transforms import RandomNodeSplit, RandomLinkSplit

from constants.constants import (
    GEOTEXT_PREPROCESSED_DATA,
    TWITTER_PREPROCESSED_DATA,
    TWITTER_GRAPH_PATH,
    device,
    seed,
)
from app.utils.utility import get_data
from app.core.graph_related.create_edges import (
    build_edges,
    build_edges_by_predicting_labels,
)
from app.core.prompt_related.llm_api import query_llm_for_node


def compute_node_properties(G, alpha=None, beta=1.0):
    """
    Compute various node properties for a given graph.
    Args:
        G (networkx.Graph): The input graph.
        alpha (float, optional): Damping factor for Katz centrality. If None, it is computed based on the graph's eigenvalues.
        beta (float, optional): Scaling factor for Katz centrality. Default is 1.0.
    Returns:
        dict: A dictionary containing various node properties:

    """
    if alpha is None:
        eigenvalues = nx.adjacency_spectrum(G)
        max_eigenvalue = max(abs(eigenvalues))
        alpha = 0.9 / max_eigenvalue  # Ensure alpha < 1/max_eigenvalue
    return {
        "square_clustering": nx.square_clustering(G),
        "clustering": nx.clustering(G),
        "degree": nx.centrality.degree_centrality(G),
        "closeness": nx.centrality.closeness_centrality(G),
        "betweenness": nx.centrality.betweenness_centrality(G),
        "katz": nx.katz_centrality(G, alpha=alpha, beta=beta),
    }


def get_one_hop_neighbors(G):
    """Get one-hop neighbors for each node"""
    neighbors = {}
    for i in tqdm.tqdm(range(G.num_nodes)):
        neighbors[i] = list(G.edge_index[1][G.edge_index[0] == i].numpy().astype(int))
    return neighbors


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
        print(f"Graph created successfully: {graph}")
        torch.save(graph, os.path.join(TWITTER_GRAPH_PATH, f"graph_{data_type}.pt"))
        return graph
    except Exception as e:
        return {"error": str(e)}


def split_graph(data_type: str, refined: bool = False, train_ratio: float = 0.8):
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
        graph_path = (
            f"graph_{data_type}_refined_nodes.pt" if refined else f"graph_{data_type}"
        )
        print(f"{graph_path} is the graph that is used to split.")
        graph_data = torch.load(
            os.path.join(TWITTER_GRAPH_PATH, graph_path),
            map_location=torch.device(device),
        )
        print(f"Graph loaded successfully from {TWITTER_GRAPH_PATH}.")
        print(f"Graph data: {graph_data}")
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Split the graph into train, validation and test sets
        splitter = RandomNodeSplit(split="train_rest", num_val=0.1, num_test=0.1)
        splitted_graph_data = splitter(graph_data)
        print(f"Graph data after splitting: {splitted_graph_data}")

        torch.save(
            splitted_graph_data,
            os.path.join(TWITTER_GRAPH_PATH, f"graph_{data_type}_splitted.pt"),
        )

        # Get text from the feature embeddings of training, validation and test set and save in csv file with labels
        node_refined_data_path = os.path.join(
            TWITTER_GRAPH_PATH, f"{data_type}_llm_query_with_results.csv"
        )
        if os.path.exists(node_refined_data_path):
            node_refined_data = get_data(node_refined_data_path)
            print(node_refined_data["answer"].tolist()[:5])
        if data_type == "twitter":
            preprocessed_data = get_data(TWITTER_PREPROCESSED_DATA)
        elif data_type == "geotext":
            preprocessed_data = get_data(GEOTEXT_PREPROCESSED_DATA)

        train_indices = splitted_graph_data.train_mask.nonzero(as_tuple=True)[0]
        val_indices = splitted_graph_data.val_mask.nonzero(as_tuple=True)[0]
        test_indices = splitted_graph_data.test_mask.nonzero(as_tuple=True)[0]
        train_texts = (
            preprocessed_data.iloc[train_indices]["feature"].tolist()
            if not refined
            else node_refined_data.iloc[train_indices]["answer"].tolist()
        )
        train_labels = preprocessed_data.iloc[train_indices]["label"].tolist()

        val_texts = (
            preprocessed_data.iloc[val_indices]["feature"].tolist()
            if not refined
            else node_refined_data.iloc[val_indices]["answer"].tolist()
        )
        val_labels = preprocessed_data.iloc[val_indices]["label"].tolist()

        test_texts = (
            preprocessed_data.iloc[test_indices]["feature"].tolist()
            if not refined
            else node_refined_data.iloc[test_indices]["answer"].tolist()
        )
        test_labels = preprocessed_data.iloc[test_indices]["label"].tolist()

        # Save the train, validation and test data in csv files
        train_df = pd.DataFrame(
            {"indices": train_indices, "feature": train_texts, "label": train_labels}
        )
        train_df.to_csv(
            os.path.join(TWITTER_GRAPH_PATH, f"graph_{data_type}_train_data.csv"),
            index=False,
        )

        val_df = pd.DataFrame(
            {"indices": val_indices, "feature": val_texts, "label": val_labels}
        )
        val_df.to_csv(
            os.path.join(TWITTER_GRAPH_PATH, f"graph_{data_type}_val_data.csv"),
            index=False,
        )

        test_df = pd.DataFrame(
            {"indices": test_indices, "feature": test_texts, "label": test_labels}
        )
        test_df.to_csv(
            os.path.join(TWITTER_GRAPH_PATH, f"graph_{data_type}_test_data.csv"),
            index=False,
        )

        return True
    except Exception as e:
        print(f"An error occurred while splitting graph: {e}")
        return False


def get_node_properties(data_type: str):
    """
    Get node properties of the graph.

    Args:
        data_type (str): Type of data to get node properties (e.g., "twitter", "geotext").

    Returns:
        dict: Node properties including features and labels.
    """
    try:
        # Load the graph
        graph_data = torch.load(
            os.path.join(TWITTER_GRAPH_PATH, f"graph_{data_type}.pt"),
            map_location=torch.device(device),
        )
        if graph_data is None:
            print(f"No graph data found for {data_type}.")
        print(f"Graph loaded successfully from {TWITTER_GRAPH_PATH}.")
        G = to_networkx(graph_data)
        topo_features = compute_node_properties(G)
        # Store topological features in a csv file
        topo_features_df = pd.DataFrame(topo_features)
        topo_features_df.to_csv(
            os.path.join(TWITTER_GRAPH_PATH, f"{data_type}_graph_topological_info.csv"),
            index=True,
        )
        print(f"Topological features saved to {TWITTER_GRAPH_PATH}.")
        return True
    except Exception as e:
        return {"error": str(e)}


def process_topological_features(data_type: str):
    try:
        """Process and rank topological features"""
        topo_features = pd.read_csv(
            os.path.join(TWITTER_GRAPH_PATH, f"{data_type}_graph_topological_info.csv"),
            index_col=0,
        )
        rename = {
            "clustering": "Clustering Coefficient",
            "degree": "Node Degree",
            "square_clustering": "Square Clustering Coefficient",
            "closeness": "Closeness Centrality",
            "betweenness": "Betweenness Centrality",
            "katz": "Katz Centrality",
        }
        topo_features = {rename[k]: v for k, v in topo_features.items()}
        # Calculate ranks
        topo_features_rank = {}
        for method, values in topo_features.items():
            sorted_values = dict(
                sorted(values.items(), key=lambda x: x[1], reverse=True)
            )
            rank = 0
            pre_value = -1
            node2rank = {}
            for node, value in sorted_values.items():
                if value != pre_value:
                    rank += 1
                    pre_value = value
                node2rank[node] = rank
            topo_features_rank[method] = node2rank

        # Restructure features
        new_topo_features = {}
        for method, values in topo_features.items():
            for node, value in values.items():
                if node not in new_topo_features:
                    new_topo_features[node] = {}
                new_topo_features[node][method] = (
                    value,
                    topo_features_rank[method][node],
                )
        topo_features_rank_df = pd.DataFrame(new_topo_features)
        topo_features_rank_df.to_csv(
            os.path.join(
                TWITTER_GRAPH_PATH, f"{data_type}_graph_topological_preprocessed.csv"
            ),
            index=True,
        )
        print(f"Preprocessed topological features saved to {TWITTER_GRAPH_PATH}.")
        return True
    except Exception as e:
        print(e)
        return {"error": str(e)}


templates = {
    "base": "Given a node from a {} graph, where the node type is {} with {} nodes, and the edge type is {} with {} edges.\n ",
    "node_text": 'The textual node description is "{}".\n ',
    "connectivity": 'One similar node has the description "{}".\n ',
    "property": 'The value of property "{}" is {:.4f}, ranked at {} among {} nodes.\n ',
    "final": "Output the potential class of the node and provide reasons for your assessment. The classes are {}. Your answer should be less than 200 words.\n ",
}

# Constants and templates
graph_stats = {
    "twitter": {
        "graph_type": "twitter network",
        "node_type": "twitter",
        "edge_type": "connectivity between genders",
        "total_node": 10750,
        "total_edge": 65074,
    },
}

classes = {
    "twitter": ["male", "female"],
}


def generate_text(node, methods, data_name, descriptions, similar_nodes):
    """Generate text description for a single node"""
    text = templates["base"].format(
        graph_stats[data_name]["graph_type"],
        graph_stats[data_name]["node_type"],
        graph_stats[data_name]["total_node"],
        graph_stats[data_name]["edge_type"],
        graph_stats[data_name]["total_edge"],
    )

    text += templates["node_text"].format(descriptions[node])
    if len(similar_nodes[node]) > 0:
        text += (
            "The following are the connectivity information of the similar nodes:\n "
        )
        for similar_node in similar_nodes[node]:
            text += templates["connectivity"].format(descriptions[similar_node])
    else:
        text += "This node has no similar nodes.\n"

    for method, (value, rank) in methods.items():
        text += templates["property"].format(
            method, value, rank, graph_stats[data_name]["total_node"]
        )

    text += templates["final"].format(classes[data_name])
    return text


def generate_llm_query_for_node_properties(data_type: str):
    try:
        # Load the graph
        graph_data = torch.load(
            os.path.join(TWITTER_GRAPH_PATH, f"graph_{data_type}.pt"),
        )
        twitter_data = pd.read_csv(TWITTER_PREPROCESSED_DATA)
        topo_features_rank = pd.read_csv(
            os.path.join(
                TWITTER_GRAPH_PATH, f"{data_type}_graph_topological_preprocessed.csv"
            ),
            index_col=0,
        )
        one_hop_neighbors = get_one_hop_neighbors(graph_data)
        features_dict = {}
        for node in topo_features_rank.columns:
            node_idx = int(node)
            features_dict[node_idx] = {}
            for feature in topo_features_rank.index:
                value_rank = topo_features_rank.loc[feature, node]
                # Convert string "(value, rank)" to tuple
                value, rank = ast.literal_eval(value_rank)
                features_dict[node_idx][feature] = (value, rank)

        texts = [
            generate_text(
                node,
                methods,
                data_name=data_type,
                descriptions=twitter_data["feature"].tolist(),
                similar_nodes=one_hop_neighbors,
            )
            for node, methods in features_dict.items()
        ]

        # Save the generated texts to a CSV file
        df = pd.DataFrame(
            {"question": texts, "answer": "Error", "node_idx": range(len(texts))}
        )

        df.to_csv(
            os.path.join(TWITTER_GRAPH_PATH, f"{data_type}_llm_query.csv"), index=True
        )
        print("LLM query generated for each node properties.")
        return True
    except Exception as e:
        print(e)
        return {"error": str(e)}


def get_llm_result_for_node(data_type: str):
    try:
        result_path = os.path.join(
            TWITTER_GRAPH_PATH, f"{data_type}_llm_query_with_results.csv"
        )
        texts = pd.read_csv(
            os.path.join(TWITTER_GRAPH_PATH, f"{data_type}_llm_query.csv")
        )["question"].tolist()
        answers = pd.read_csv(result_path)["answer"].tolist()
        answers_df = pd.DataFrame(
            {"question": texts, "answer": answers, "node_idx": range(len(texts))}
        )
        for i in range(0, len(texts)):
            try:
                print(f"Processing text {i}/{len(texts)}")
                response = query_llm_for_node(texts[i], model="gemma3:latest")
                # Save the response in csv
                answers_df.at[i, "answer"] = response
                answers_df.to_csv(result_path, index=False)
            except Exception as e:
                print("An error occured", e)
                answers_df.at[i, "answer"] = "Error"
                answers_df.to_csv(result_path, index=False)
                print("LLM query generated for each node properties.")
                return True
    except Exception as e:
        print(e)
        return {"error": str(e)}


def generate_graph_with_new_features(data_type: str):
    """
    Create a graph based on the specified data type.

    Args:
        data_type (str): Type of data to create graph (e.g., "twitter", "geotext").
    """
    try:
        # Load the graph
        graph_data = torch.load(
            os.path.join(TWITTER_GRAPH_PATH, f"graph_{data_type}.pt"),
            map_location=torch.device(device),
        )
        print(f"Graph loaded successfully from {TWITTER_GRAPH_PATH}.")
        llm_answers = pd.read_csv(
            os.path.join(TWITTER_GRAPH_PATH, f"{data_type}_llm_query_with_results.csv")
        )["answer"].to_list()
        feature_embeddings = get_feature_embeddings(llm_answers)
        graph_data.x = feature_embeddings
        print(graph_data)
        # Save the updated graph
        updated_graph_path = os.path.join(
            TWITTER_GRAPH_PATH, f"graph_{data_type}_refined_nodes.pt"
        )
        torch.save(graph_data, updated_graph_path)
        print(
            f"Graph updated with new feature embeddings and saved to {updated_graph_path}."
        )
        return True
    except Exception as e:
        print(e)
        return {"error": str(e)}

def create_custom_shot_train_mask(graph_data, num_shots=1):
    """
    Create a custom shot train mask for the graph data.

    Args:
        graph_data (Data): The graph data object containing node features and labels.
        num_shots (int): The number of shots (examples) to use for each class.

    Returns:
        Data: The updated graph data with a custom shot train mask.
    """
    y = graph_data.y.squeeze().cpu().numpy()  # Get labels as numpy array
    classes = set(y)  # Unique class labels
    train_mask = graph_data.train_mask.clone()  # Clone the existing train mask
    # I need the index of the nodes in the train mask that are true
    true_indices = train_mask.nonzero().cpu().numpy().flatten()
    train_node_with_classes = []
    for each_node in true_indices:
        train_node_with_classes.append(
            {
                "node_idx": each_node,
                "class": y[each_node],
            }
        )
    # Create a new train mask with the selected nodes
    new_train_mask = torch.zeros_like(graph_data.train_mask)
    for c in classes:
        count = 0
        for item in train_node_with_classes:
            if count >= num_shots:
                break
            if item["class"] == c:
                new_train_mask[item["node_idx"]] = True
                count += 1
    print(
        f"Number of true values in new train mask: {new_train_mask.sum().item()}"
    ) 
    graph_data.train_mask = new_train_mask
    return graph_data
