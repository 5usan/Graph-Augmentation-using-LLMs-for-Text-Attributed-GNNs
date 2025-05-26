from fastapi import APIRouter
from ..utils.preprocess_data import preprocess_data

router = APIRouter()


@router.get("/pre-process_data")
def preprocess_data_endpoint(data_type: str):
    """
    Endpoint to preprocess data.
    Args:
        data_type (str): Type of data to preprocess (e.g., "twitter", "geotext").
    """
    try:
        result = preprocess_data(data_type)
        return_value = (
            {"message": f"{data_type} data preprocessed successfully."}
            if result
            else {"message": f"Failed to preprocess {data_type} data."}
        )
        return return_value
    except Exception as e:
        return {"error": str(e)}
