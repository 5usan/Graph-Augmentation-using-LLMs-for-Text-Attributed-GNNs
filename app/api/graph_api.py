from fastapi import APIRouter
from app.scripts.train_eval import train_eval_model
from app.core.graph_related.graph import (
    generate_graph,
    split_graph,
    get_node_properties,
    process_topological_features,
    generate_llm_query_for_node_properties,
    get_llm_result_for_node,
    generate_graph_with_new_features,
)

router = APIRouter()


@router.get("/create_graph")
def create_graph_endpoint(data_type: str):
    """
    Endpoint to create a graph.
     Args:
        data_type (str): Type of data to create graph (e.g., "twitter", "geotext").
    """
    try:
        # Placeholder for graph creation logic
        if data_type not in ["twitter", "geotext"]:
            return {"error": "Invalid data type. Choose either 'twitter' or 'geotext'."}
        graph = generate_graph(data_type)
        return {"message": "Graph created successfully."}
    except Exception as e:
        return {"error": str(e)}


@router.get("/get_node_properties")
def get_node_properties_endpoint(data_type: str):
    """
    Endpoint to get node properties of a graph.
     Args:
        data_type (str): Type of data to get node properties (e.g., "twitter", "geotext").
    """
    try:
        # Placeholder for getting node properties logic
        if data_type not in ["twitter", "geotext"]:
            return {"error": "Invalid data type. Choose either 'twitter' or 'geotext'."}
        # Call the function to get node properties
        result = get_node_properties(data_type)
        return (
            {"message": "Node properties retrieved successfully."}
            if result == True
            else {"error": "Some error occurred while retrieving node properties."}
        )
    except Exception as e:
        return {"error": str(e)}


@router.get("/preprocess_graph_topological_features")
def preprocess_graph_topological_features_endpoint(data_type: str):
    """
    Endpoint to get preprocess topological info of a graph.
     Args:
        data_type (str): Type of data to get node properties (e.g., "twitter", "geotext").
    """
    try:
        # Placeholder for getting node properties logic
        if data_type not in ["twitter", "geotext"]:
            return {"error": "Invalid data type. Choose either 'twitter' or 'geotext'."}
        # Call the function to get node properties
        result = process_topological_features(data_type)
        return (
            {"message": "Node properties retrieved successfully."}
            if result == True
            else {"error": "Some error occurred while preprocessing node properties."}
        )
    except Exception as e:
        return {"error": str(e)}


@router.get("/generate_llm_query_for_nodes")
def generate_llm_query_for_nodes_endpoint(data_type: str):
    """
    Endpoint to get preprocess topological info of a graph.
     Args:
        data_type (str): Type of data to get node properties (e.g., "twitter", "geotext").
    """
    try:
        # Placeholder for getting node properties logic
        if data_type not in ["twitter", "geotext"]:
            return {"error": "Invalid data type. Choose either 'twitter' or 'geotext'."}
        # Call the function to get node properties
        result = generate_llm_query_for_node_properties(data_type)
        return (
            {"message": "LLM query for node properties generated successfully."}
            if result == True
            else {
                "error": "Some error occurred while generating llm query for node properties."
            }
        )
    except Exception as e:
        return {"error": str(e)}


@router.get("/get_llm_result_for_nodes")
def get_llm_result_for_nodes_endpoint(data_type: str):
    """
    Endpoint to get preprocess topological info of a graph.
     Args:
        data_type (str): Type of data to get node properties (e.g., "twitter", "geotext").
    """
    try:
        # Placeholder for getting node properties logic
        if data_type not in ["twitter", "geotext"]:
            return {"error": "Invalid data type. Choose either 'twitter' or 'geotext'."}
        # Call the function to get node properties
        result = get_llm_result_for_node(data_type)
        return (
            {"message": "LLM results for node properties generated successfully."}
            if result == True
            else {
                "error": "Some error occurred while generating llm result for node properties."
            }
        )
    except Exception as e:
        return {"error": str(e)}


@router.get("/create_graph_with_refine_nodes")
def create_graph_with_refine_nodes_endpoint(data_type: str):
    """
    Endpoint to create a graph.
     Args:
        data_type (str): Type of data to create graph (e.g., "twitter", "geotext").
    """
    try:
        # Placeholder for graph creation logic
        if data_type not in ["twitter", "geotext"]:
            return {"error": "Invalid data type. Choose either 'twitter' or 'geotext'."}
        graph = generate_graph_with_new_features(data_type)
        return {"message": "Graph with new features created successfully."}
    except Exception as e:
        return {"error": str(e)}


@router.get("/split_graph")
def split_graph_endpoint(data_type: str, refined: bool = False):
    """
    Endpoint to split a graph.
     Args:
        data_type (str): Type of data to create graph (e.g., "twitter", "geotext").
    """
    try:
        # Placeholder for graph splitting logic
        if data_type not in ["twitter", "geotext"]:
            return {"error": "Invalid data type. Choose either 'twitter' or 'geotext'."}
        # Call the function to split the graph
        split_graph(data_type, refined)
        return {"message": "Graph split successfully."}
    except Exception as e:
        return {"error": str(e)}


@router.get("/train_eval_model")
def train_eval_model_endpoint(
    data_type: str,
    model_type: str = "gcn",
    epochs: int = 200,
    learning_rate: float = 0.01,
):
    """
    Endpoint to train and evaluate the GCN model.
     Args:
        data_type (str): Type of graph (twitter, geotext).
        model_type (str): Type of model to use (currently "gcn" and "gat" is supported).
    """
    try:
        # Placeholder for training and evaluation logic
        if data_type not in ["twitter", "geotext"]:
            return {"error": "Invalid data type. Choose either 'twitter' or 'geotext'."}
        if model_type not in ["gcn", "gat"]:
            return {
                "error": "Invalid model type. Supported types are 'gcn' and 'gat'. Defaulting to 'gcn'."
            }

        train_eval_model(data_type, model_type, epochs, learning_rate)
        return {"message": "Model trained and evaluated successfully."}
    except Exception as e:
        print(e)
        return {"error": str(e)}
