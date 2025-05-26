import re
import contractions
import unicodedata
import pandas as pd


def get_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the raw data.
    """
    try:
        data = pd.read_csv(file_path, encoding="ISO-8859-1")
        print(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return pd.DataFrame()  # Return an empty DataFrame if the file is not found
    except pd.errors.EmptyDataError:
        print(f"No data: {file_path} is empty")
        return pd.DataFrame()  # Return an empty DataFrame if the file is empty
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()  # Return an empty DataFrame for any other exceptions


def text_regularization(text: str) -> str:
    """
    Perform text regularization on the input text.

    Args:
        text (str): Inpput text to be regualarized

    Returns:
        str: Regularized text.
    """
    try:
        text = str(text)
        # Convert to lowercase
        text = text.lower()

        # Expand contractions, can't => cannot
        text = contractions.fix(text)

        # # Remove punctuations
        # text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove special characters
        text = re.sub(r"[^a-z\s]", "", text)

        # Normalize accented characters "café" → "cafe"
        text = (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("utf-8", "ignore")
        )

        # Remove extra white spaces
        text = " ".join(text.split())
        return text if len(text) > 0 else None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def create_csv(dataframe, filename):
    """
    Create a CSV file from a DataFrame.

    Args:
        dataframe (pd.DataFrame): DataFrame to save as CSV.
        filename (str): Name of the CSV file to save.
    """
    try:
        dataframe.to_csv(filename, index=False)
        print(f"CSV file {filename} created successfully.")
    except Exception as e:
        print(f"An error occurred while creating CSV: {e}")
