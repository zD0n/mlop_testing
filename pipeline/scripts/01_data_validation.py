import pandas as pd
import mlflow
import kagglehub
import shutil
import os
def validate_data():

    mlflow.set_experiment("Emotion classification - Data Validation")


    with mlflow.start_run():
        print("Starting data validation run...")
        mlflow.set_tag("ml.step", "data_validation")

        path = kagglehub.dataset_download("praveengovi/emotions-dataset-for-nlp")

        dataset = "dataset"
        os.makedirs(dataset, exist_ok=True)
        target_path = f"./{dataset}"
        
        shutil.copytree(path, target_path, dirs_exist_ok=True)

        df = pd.read_csv("./dataset/train.txt",
                        delimiter=';', header=None, names=['sentence','label'])

        val_df = pd.read_csv("./dataset/val.txt",
                        delimiter=';', header=None, names=['sentence','label'])

        ts_df = pd.read_csv("./dataset/test.txt",
                        delimiter=';', header=None, names=['sentence','label'])

        new_df = pd.concat([df,val_df,ts_df],axis=0)

        num_rows, num_cols = new_df.shape
        num_classes = new_df['label'].nunique()
        missing_values = new_df.isnull().sum().sum()

        print(f"Dataset shape: {num_rows} rows, {num_cols} columns")
        print(f"Number of classes: {num_classes}")
        print(f"Missing values: {missing_values}")

        mlflow.log_metric("num_rows", num_rows)
        mlflow.log_metric("num_cols", num_cols)
        mlflow.log_metric("missing_values", missing_values)
        mlflow.log_param("num_classes", num_classes)

        validation_status = "Success"
        if missing_values > 0 or num_classes < 2:
            validation_status = "Failed"

        mlflow.log_param("validation_status", validation_status)
        print(f"Validation status: {validation_status}")
        print("Data validation run finished.")


if __name__ == "__main__":
    validate_data()
