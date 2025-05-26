import json
import ollama


def predict_label(data, feature_index_1, model="deepseek-r1:8b", label="gender"):
    """
    Given a text, classify whether it was likely written by one of the {label}.

    Args:
        data (dict): Dictionary containing the features and labels.
        feature_index_1 (int): Index of the first feature to classify.
        model (str): Model name for the LLM API.
        label (str): Label to consider for classification.

    Returns:
        one of the labels: The predicted label for the text.

    """
    try:
        return_value = {"label": "", "confidence": "", "explaination": ""}
        response_format = {
            "gender": "<male or female>",
            "confidence": f"<float between 0 and 1, where 1 means highly related but give value in string format>",
            # "explaination": f"<ONLY one sentence that explains how the two texts are related according to the {label} and which {label} they are related to>",
        }
        text_1 = data["feature"][feature_index_1]
        print(f"Text A: {text_1}, Label:{data['label'][feature_index_1]}")
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": f"""
                    You are an expert in sociolinguistic analysis. Given a tweet, your task is to classify whether it was likely written by one of the {label}. Focus only on the following linguistic aspects:
                            - Sentence structure and syntax
                            - Use of personal pronouns or emotional expressions
                            - Punctuation, slang, and informal tones
                            - Communication style (e.g., assertive, expressive, reserved, playful)
                            - Writing rhythm or stylistic patterns
                        Do not base your classification on topic alone (e.g., sports, art, romance, tech) or rely on stereotypes.
                        Text: {text_1}
                        Return your answer in structured JSON format only.
                        The JSON response MUST include both `related` and `similarity_score` fields. JSON format: {response_format}
                        """,
                },
            ],
            keep_alive="5m",
            format="json",
            stream=False,
            options={"temperature": 0},
        )
        content = response["message"]["content"]
        print(content)
        data = json.loads(content)
        if bool(data):
            return_value["label"] = data["gender"]
            return_value["confidence"] = str(data["confidence"])
            # return_value['explaination'] = str(data["explaination"])
        return return_value
    except Exception as e:
        print("An error occured while predicting labels", {e})
