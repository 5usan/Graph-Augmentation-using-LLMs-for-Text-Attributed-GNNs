import os
import pandas as pd

from app.utils.utility import get_data, text_regularization
from constants.constants import (
    GEOTEXT_PREPROCESSED_DATA,
    GEOTEXT_RAW_DATA,
    TWITTER_PREPROCESSED_DATA,
    TWITTER_RAW_DATA,
)

def preprocess_twitter_data(
    dataset_path: str, preprocessed_dataset_path
) -> pd.DataFrame:
    """
    Preprocess the dataset by loading it and applying text regularization. After preprocessing, data is saved to a csv file.

    Args:
        dataset_path (str): Path to the dataset CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the preprocessed data having feature and label columns.
    """
    try:
        twitter_raw_data = get_data(dataset_path)
        if twitter_raw_data.empty:
            print("No data to preprocess.")
            return None
        twitter_data = twitter_raw_data[twitter_raw_data["gender:confidence"] >= 0.5]
        columns_to_keep = ["gender", "description"]
        columns_to_drop = [col for col in twitter_data if col not in columns_to_keep]
        twitter_data = twitter_data.drop(columns=columns_to_drop)
        twitter_data = twitter_data.dropna()
        twitter_data = twitter_data[twitter_data["gender"].isin(["male", "female"])]
        twitter_data = twitter_data.dropna()
        feature = twitter_data["description"]
        feature = [text_regularization(each) for each in feature]
        label = twitter_data["gender"].to_list()
        df = pd.DataFrame(data={"feature": feature, "label": label})
        df = df.dropna()
        df.to_csv(preprocessed_dataset_path)
        return True
    except Exception as e:
        print("An error occured", {e})
        print("Unabe to preprocess the dataset")
        return None


def preprocess_data(data_type: str):
    """
    Preprocess the dataset by loading it and applying text regularization. After preprocessing, data is saved to a csv file.

    Args:
        data_type (str): which dataset to preprocess, either twitter or geotext.
    """
    try:
        if data_type == "twitter":
            if not os.path.exists(TWITTER_PREPROCESSED_DATA):
                preprocess_twitter_data(TWITTER_RAW_DATA, TWITTER_PREPROCESSED_DATA)
                print("Preprocessing completed.")
            else:
                print(
                    "Preprocessed Twitter data already exists. Skipping preprocessing."
                )
                return True
        elif data_type == "geotext":
            if not os.path.exists(GEOTEXT_PREPROCESSED_DATA):
                print("Preprocessing Twitter data...")
                preprocess_twitter_data(GEOTEXT_RAW_DATA, GEOTEXT_PREPROCESSED_DATA)
                print("Preprocessing completed.")
            else:
                print(
                    "Preprocessed GeoText data already exists. Skipping preprocessing."
                )
                return True
        else:
            raise ValueError(
                "Invalid dataset specified. Choose either 'twitter' or 'geotext'."
            )

    except Exception as e:
        print(f"An error occurred while preprocessing data: {e}")
        return None
