
import mlflow
import os
import pandas as pd
import json
import string

def load_and_predict():
    """
    Simulates a production scenario by loading a model from a specific
    stage in the MLflow Model Registry and using it for prediction.
    """
    MODEL_NAME = "emotion-classifier-prod"
    MODEL_STAGE = "Staging" # Change to "Production" after transitioning the model stage


    print(f"Loading model '{MODEL_NAME}' from stage '{MODEL_STAGE}'...")
    
    # Load the model from the Model Registry
    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    except mlflow.exceptions.MlflowException as e:
        print(f"\nError loading model: {e}")
        print(f"Please make sure a model version is in the '{MODEL_STAGE}' stage in the MLflow UI.")
        return

    # ===========================
    # Load Vocabulary and Label Mapping
    # ===========================
    with open(r'.\processed_data\vocab.json', 'r', encoding='utf-8') as f:
        voc = json.load(f)['idx2word']

    # Reverse mapping for encoding
    word2idx = {v: int(k) for k, v in voc.items()}

    with open(r'.\processed_data\label_mapping.json', 'r', encoding='utf-8') as f:
        act_class = json.load(f)

    def encode_and_pad(text, max_len=100):
        tokens = [str(word2idx.get(w, word2idx.get("UNK", 0))) for w in text.split()]
        if len(tokens) < max_len:
            tokens += ["0"] * (max_len - len(tokens))  # pad with zeros
        else:
            tokens = tokens[:max_len]  # truncate if too long
        return " ".join(tokens)

    def preprocess_text(text):

        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())

        return text

    text = "I feel so alive."

    encoded_text = encode_and_pad(preprocess_text(text))

    prediction = model.predict([encoded_text])
    print(prediction)
    print("-" * 30)
    print(f"Input Text:\n{text}")
    print(f"Predicted Label: {act_class.get(str(prediction[0]))}")


if __name__ == "__main__":
    load_and_predict()


