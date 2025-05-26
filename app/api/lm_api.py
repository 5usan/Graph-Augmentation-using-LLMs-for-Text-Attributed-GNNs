from fastapi import APIRouter
from app.core.lm_related.lm_data import split_data, create_data
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
def create_data_endpoint(data_type: str):
    """
    Endpoint to create a dataset by splitting the data into training, validation, and test sets.
    
    Args:
        data_type (str): Type of data to create (e.g., "twitter", "geotext").
    """
    try:
        # Placeholder for dataset creation logic
        if data_type not in ["twitter", "geotext"]:
            return {"error": "Invalid data type. Choose either 'twitter' or 'geotext'."}

        create_data(data_type)
        return {"message": "Data created successfully."}
    except Exception as e:
        return {"error": str(e)}