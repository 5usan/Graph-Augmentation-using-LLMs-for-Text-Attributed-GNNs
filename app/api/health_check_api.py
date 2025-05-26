from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check():
    """
    Health check endpoint to verify if the API is running.
    """
    try:
        # Perform a simple check, e.g., database connection or any other service
        # For now, we just return a success message
        return {"status": "OK", "message": "API is running!"}
    except Exception as e:
        # If any error occurs, return a failure response
        return {"status": "ERROR", "message": str(e)}
