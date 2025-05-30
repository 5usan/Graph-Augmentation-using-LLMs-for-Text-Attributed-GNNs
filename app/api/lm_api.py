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
def create_data_endpoint(data_type: str, model: str = "bert"):
    """
    Endpoint to create a dataset by splitting the data into training, validation, and test sets.
    
    Args:
        data_type (str): Type of data to create (e.g., "twitter", "geotext").
        model (str): The model to be used for data creation (default is "bert").
    """
    try:
        # Placeholder for dataset creation logic
        if data_type not in ["twitter", "geotext"]:
            return {"error": "Invalid data type. Choose either 'twitter' or 'geotext'."}
        
        if model not in ["bert", "distillbert", "roberta"]:
            return {"error": "Invalid model type. Supported types are 'bert', 'distillbert', and 'roberta'."}

        create_data(data_type, model)
        return {"message": "Data created successfully."}
    except Exception as e:
        return {"error": str(e)}

@router.get("/train_eval_lm")
def train_eval_lm_endpoint(data_type: str, model: str = "bert", learning_rate: float = 1e-5, epochs: int = 10):
    """
    Endpoint to train and evaluate a language model.
    
    Args:
        model (str): The model to be trained (default is "bert").
        learning_rate (float): Learning rate for training (default is 1e-5).
        epochs (int): Number of epochs for training (default is 10).
    """
    try:
        if data_type not in ["twitter", "geotext"]:
            return {"error": "Invalid data type. Choose either 'twitter' or 'geotext'."}
        
        if model not in ["bert", "distillbert", "roberta"]:
            return {"error": "Invalid model type. Supported types are 'bert', 'distillbert', and 'roberta'."}
        # Placeholder for training and evaluation logic
        train(data_type, model, learning_rate, epochs)
        return {"message": f"Training and evaluation completed for {model}."}
    except Exception as e:
        return {"error": str(e)}