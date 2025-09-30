import sys
import os
import pandas as pd
import numpy as np
import mlflow
from mlflow.artifacts import download_artifacts
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes,
                 num_filters=100, filter_sizes=[3, 4, 5], dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k)
            for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))
            pooled = F.max_pool1d(conv_out, conv_out.size(2))
            conv_outputs.append(pooled.squeeze(2))
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        return self.fc(x)

class TorchTextClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, vocab_size=10000, embedding_dim=128, num_classes=6,
                 num_filters=100, filter_sizes=[3, 4, 5], dropout=0.5,
                 lr=0.001, batch_size=32, epochs=5, device=None):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model_ = None

    def fit(self, X, y):
        dataset = torch.utils.data.TensorDataset(
            torch.tensor([list(map(int, seq.split())) for seq in X], dtype=torch.long),
            torch.tensor(y.values if hasattr(y, "values") else y, dtype=torch.long)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_ = TextCNN(self.vocab_size, self.embedding_dim,
                              self.num_classes, self.num_filters,
                              self.filter_sizes, self.dropout).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            self.model_.train()
            total_loss, correct, total = 0, 0, 0
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model_(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
            print(f"Epoch {epoch+1}: Loss {total_loss/len(loader):.4f}, Acc {100*correct/total:.2f}%")
        return self

    def predict(self, X):
        self.model_.eval()
        dataset = torch.tensor([list(map(int, seq.split())) for seq in X], dtype=torch.long)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                output = self.model_(data)
                preds.extend(output.argmax(dim=1).cpu().numpy())
        return np.array(preds)

    def predict_proba(self, X):
        self.model_.eval()
        dataset = torch.tensor([list(map(int, seq.split())) for seq in X], dtype=torch.long)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        probs = []
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                output = torch.softmax(self.model_(data), dim=1)
                probs.extend(output.cpu().numpy())
        return np.array(probs)

class IdentityTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X

def train_evaluate_register(preprocessing_run_id,model_name,epochs=10):

    ACCURACY_THRESHOLD=0.9

    mlflow.set_experiment("Emotion classification - Model Training")

    with mlflow.start_run(run_name=f"Text CNN"):
        print(f"Starting training run with Text CNN...")
        mlflow.set_tag("ml.step", "model_training_evaluation")
        mlflow.log_param("preprocessing_run_id", preprocessing_run_id)

        try:
            local_artifact_path = download_artifacts(
                run_id=preprocessing_run_id,
                artifact_path="processed_data"
            )

            train_df = pd.read_csv(os.path.join(local_artifact_path, "train.csv"))
            test_df = pd.read_csv(os.path.join(local_artifact_path, "test.csv"))

            print("Successfully loaded data from downloaded artifacts.")
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            sys.exit(1)

        pipeline = Pipeline([
            ("identity", IdentityTransformer()),
            ("model", TorchTextClassifier(epochs=epochs, batch_size=32))
        ])

        pipeline.fit(train_df["sequence"], train_df["label"])

        y_pred = pipeline.predict(test_df["sequence"])
        acc = accuracy_score(test_df["label"], y_pred)
        print(f"Test Accuracy: {acc:.4f}")

        mlflow.log_metric("accuracy", acc)

        if acc >= ACCURACY_THRESHOLD:
            print(f"Model accuracy {acc:.4f} meets the threshold. Registering model...")
            mlflow.sklearn.log_model(pipeline, "Emotion_classification_pipeline")
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/Emotion_classification_pipeline"
            registered_model = mlflow.register_model(model_uri, model_name)
            print(f"Model registered as '{registered_model.name}' version {registered_model.version}")
            
            with open(f"{model_name}.pkl", "wb") as f:
                pickle.dump(pipeline, f)
            print(f"Pipeline saved to {model_name}.pkl")
        else:
            print(f"Model accuracy {acc:.4f} is below the threshold. Not registering.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/03_train_evaluate_register.py <preprocessing_run_id> <epochs> <Model Name>")
        sys.exit(1)
    
    run_id = sys.argv[1]
    epochs = sys.argv[2]
    name = sys.argv[3]
    train_evaluate_register(preprocessing_run_id=run_id,epochs=int(epochs),model_name=name)
