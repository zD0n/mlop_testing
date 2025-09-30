
import mlflow
import os
import pandas as pd
import json
import string
import mlflow
import sys
from mlflow.tracking import MlflowClient
from mlflow.tracking import MlflowClient



def encode_and_pad(text,word2idx, max_len=100):
    tokens = [str(word2idx.get(w, word2idx.get("UNK", 0))) for w in text.split()]
    if len(tokens) < max_len:
        tokens += ["0"] * (max_len - len(tokens)) 
    else:
        tokens = tokens[:max_len]
    return " ".join(tokens)

def preprocess_text(text):

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())

    return text

def load_and_predict(model_name,text):

    # Load the model from the Model Registry
    try:
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        latest_version = max(versions, key=lambda v: int(v.version))

        model = mlflow.pyfunc.load_model(f"models:/{model_name}/{latest_version.version}")

    except mlflow.exceptions.MlflowException as e:
        print(f"\nError loading model: {e}")
        sys.exit(1)

    with open(r'.\processed_data\vocab.json', 'r', encoding='utf-8') as f:
        voc = json.load(f)['idx2word']

    word2idx = {v: int(k) for k, v in voc.items()}

    with open(r'.\processed_data\label_mapping.json', 'r', encoding='utf-8') as f:
        act_class = json.load(f)

    encoded_text = encode_and_pad(preprocess_text(text),word2idx)

    prediction = model.predict([encoded_text])
    print("-" * 30)
    print(f"Input Text:\n{text}")
    print(f"Predicted Label: {act_class.get(str(prediction[0]))}")
    print("-" * 30)


if __name__ == "__main__":
    try:
        if len(sys.argv) != 3:
            print("Usage: python scripts/04_transition_model.py <model_name> <Text Eg. 'I Feel so alive.'>")
            print("Or you can")
            sys.exit(1)

        model_name_arg = sys.argv[1]
        input_text = sys.argv[2]
        load_and_predict(model_name_arg, input_text)
    except:
        try:
            client = MlflowClient()

            # Get all registered models
            registered_models = client.search_registered_models()
            print("-" * 30)
            print("Registered models:")
            for model in registered_models:
                print("-", model.name)
            print("-" * 30)
            model_name_arg= str(input("Model Name: "))
            input_text = str(input("Input Text: "))
            load_and_predict(model_name_arg,input_text)
        except:
            print("try train some model first. Incase if you already train maybe it about the path")
