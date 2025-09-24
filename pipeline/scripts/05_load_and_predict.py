from sklearn.datasets import load_breast_cancer
import mlflow
import os
import pandas as pd
import json

def load_and_predict():
    """
    Simulates a production scenario by loading a model from a specific
    stage in the MLflow Model Registry and using it for prediction.
    """
    MODEL_NAME = "emotion-classification-prod"
    MODEL_STAGE = "Staging" # Change to "Production" after transitioning the model stage


    print(f"Loading model '{MODEL_NAME}' from stage '{MODEL_STAGE}'...")
    
    # Load the model from the Model Registry
    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    except mlflow.exceptions.MlflowException as e:
        print(f"\nError loading model: {e}")
        print(f"Please make sure a model version is in the '{MODEL_STAGE}' stage in the MLflow UI.")
        return

    test_val = pd.read_csv("S:/MLOPs/test.csv")
    X = test_val['sequence']
    y = test_val['label']

    with open('S:/MLOPs/vocab.json', 'r', encoding='utf-8') as f:
        voc = json.load(f)['idx2word']
    with open('S:/MLOPs/label_mapping.json', 'r', encoding='utf-8') as f:
        act_class = json.load(f)
        
    decoding  = lambda x: " ".join([voc.get(i, "UNK") if i != "0" else "" for i in x.split() ])
    
    sample_data = X[0:1].to_list()

    actual_label = y[0]
    
    prediction = model.predict(sample_data)
    text = input("Enter a text: ")

    print("-" * 30)
    print(f"Sample Data Features:\n{decoding(sample_data[0]).strip()}")
    print(f"Actual Label: {act_class.get(str(actual_label))}")
    print(f"Predicted Label: {act_class.get(str(prediction[0]))}")



if __name__ == "__main__":
    load_and_predict()
