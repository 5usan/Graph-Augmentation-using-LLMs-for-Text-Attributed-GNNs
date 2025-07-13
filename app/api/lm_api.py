from fastapi import APIRouter
from app.core.lm_related.lm_data import split_data, create_data
from app.core.lm_related.trainer import train

router = APIRouter()


@router.get("/split_data")
def split_data_endpoint(data_type: str):
    """
    Endpoint to split data into training and testing sets.

    Args:
        data_type (str): Type of data to split (e.g., "twitter", "geotext").
    """
    try:
        # Placeholder for data splitting logic
        if data_type not in ["twitter", "geotext"]:
            return {"error": "Invalid data type. Choose either 'twitter' or 'geotext'."}

        split_data(data_type)
        return {"message": "Data split successfully."}
    except Exception as e:
        return {"error": str(e)}


@router.get("/create_data")
def create_data_endpoint(data_type: str, model: str = "bert", few_shot: bool = False, number_of_shots: int = 5):
    """
    Endpoint to create a dataset by splitting the data into training, validation, and test sets.

    Args:
        data_type (str): Type of data to create (e.g., "twitter", "geotext").
        model (str): The model to be used for data creation (default is "bert").
        few_shot (bool): Whether to use few-shot learning (default is False).
        number_of_shots (int): Number of shots for few-shot learning (default is
    """
    try:
        # Placeholder for dataset creation logic
        if data_type not in ["twitter", "geotext"]:
            return {"error": "Invalid data type. Choose either 'twitter' or 'geotext'."}

        if model not in ["bert", "distillbert", "roberta", "MLP"]:
            return {
                "error": "Invalid model type. Supported types are 'bert', 'distillbert', 'roberta' and 'MLP'."
            }

        create_data(data_type, model, few_shot, number_of_shots)
        return {"message": "Data created successfully."}
    except Exception as e:
        return {"error": str(e)}


@router.get("/train_eval_lm")
def train_eval_lm_endpoint(
    data_type: str,
    model: str = "bert",
    learning_rate: float = 1e-5,
    epochs: int = 10,
    preTrained: bool = False,
):
    """
    Endpoint to train and evaluate a language model.

    Args:
        model (str): The model to be trained (default is "bert").
        learning_rate (float): Learning rate for training (default is 1e-5).
        epochs (int): Number of epochs for training (default is 10).
        preTrained (bool): Whether to use a pre-trained model (default is False).
        data_type (str): Type of data to train on (e.g., "twitter", "geotext").
    """
    try:
        if data_type not in ["twitter", "geotext"]:
            return {"error": "Invalid data type. Choose either 'twitter' or 'geotext'."}

        if model not in ["bert", "distillbert", "roberta", "MLP"]:
            return {
                "error": "Invalid model type. Supported types are 'bert', 'distillbert', 'roberta' and 'MLP'."
            }
        # Placeholder for training and evaluation logic
        train(data_type, model, learning_rate, epochs, preTrained)
        return {"message": f"Training and evaluation completed for {model}."}
    except Exception as e:
        return {"error": str(e)}


@router.get("/train_mlp")
def train_mlp_endpoint(
    data_type: str, learning_rate: float = 1e-5, epochs: int = 10, refined: bool = False
):
    """
    Endpoint to train a multi-layer perceptron (MLP) model.

    Args:
        model (str): The model to be trained (default is "bert").
        learning_rate (float): Learning rate for training (default is 1e-5).
        epochs (int): Number of epochs for training (default is 10).
    """
    try:
        if data_type not in ["twitter", "geotext"]:
            return {"error": "Invalid data type. Choose either 'twitter' or 'geotext'."}

        # Placeholder for MLP training logic
        train(data_type, "MLP", learning_rate, epochs, refined)
        return {"message": f"MLP training completed for {data_type}."}
    except Exception as e:
        return {"error": str(e)}
