import os
import pandas as pd
import mlflow
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import string
import json

def preprocess_data():

    mlflow.set_experiment("Emotion classification - Data Preprocessing")


    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Starting data preprocessing run with run_id: {run_id}")
        mlflow.set_tag("ml.step", "data_preprocessing")

        df = pd.read_csv("./dataset/train.txt",
                        delimiter=';', header=None, names=['sentence','label'])

        val_df = pd.read_csv("./dataset/val.txt",
                        delimiter=';', header=None, names=['sentence','label'])

        ts_df = pd.read_csv("./dataset/test.txt",
                        delimiter=';', header=None, names=['sentence','label'])

        def preprocess_text(text):

            text = text.lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = ' '.join(text.split())

            return text
        
        df['sentence'] = df['sentence'].apply(preprocess_text)
        val_df['sentence'] = val_df['sentence'].apply(preprocess_text)
        ts_df['sentence'] = ts_df['sentence'].apply(preprocess_text)


        class Vocabulary:
            def __init__(self, max_vocab_size=10000):
                self.word2idx = {'<PAD>': 0, '<UNK>': 1}
                self.idx2word = {0: '<PAD>', 1: '<UNK>'}
                self.max_vocab_size = max_vocab_size

            def build_vocab(self, texts):
                word_counts = Counter()
                for text in texts:
                    words = text.split()
                    word_counts.update(words)

                most_common = word_counts.most_common(self.max_vocab_size - 2)

                for i, (word, _) in enumerate(most_common):
                    self.word2idx[word] = i + 2
                    self.idx2word[i + 2] = word

            def text_to_sequence(self, text, max_length=100):
                words = text.split()
                sequence = [self.word2idx.get(word, 1) for word in words]  # 1 is <UNK>

                if len(sequence) < max_length:
                    sequence.extend([0] * (max_length - len(sequence)))  # 0 is <PAD>
                else:
                    sequence = sequence[:max_length]

                return sequence

            def __len__(self):
                return len(self.word2idx)

        vocab = Vocabulary(max_vocab_size=10000)
        vocab.build_vocab(df['sentence'].tolist())

        MAX_LENGTH = 100

        X_train = [vocab.text_to_sequence(text, MAX_LENGTH) for text in df['sentence']]
        X_val = [vocab.text_to_sequence(text, MAX_LENGTH) for text in val_df['sentence']]
        X_test = [vocab.text_to_sequence(text, MAX_LENGTH) for text in ts_df['sentence']]

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(df['label'])
        y_val = label_encoder.transform(val_df['label'])
        y_test = label_encoder.transform(ts_df['label'])

        X_train_np = np.array(X_train)
        X_val_np = np.array(X_val)
        X_test_np = np.array(X_test)

        y_train_np = np.array(y_train)
        y_val_np = np.array(y_val)
        y_test_np = np.array(y_test)
        mapping = {int(i): cls for i, cls in enumerate(label_encoder.classes_)}

        train_df = pd.DataFrame({
            "sequence": [" ".join(map(str, seq)) for seq in X_train_np],
            "label": y_train_np
        })

        val_df = pd.DataFrame({
            "sequence": [" ".join(map(str, seq)) for seq in X_val_np],
            "label": y_val_np
        })

        test_df = pd.DataFrame({
            "sequence": [" ".join(map(str, seq)) for seq in X_test_np],
            "label": y_test_np
        })

        processed_data_dir = "processed_data"
        os.makedirs(processed_data_dir, exist_ok=True)

        with open("./processed_data/vocab.json", "w", encoding="utf-8") as f:
            json.dump({
                "word2idx": vocab.word2idx,
                "idx2word": vocab.idx2word,
                "max_vocab_size": vocab.max_vocab_size
            }, f, ensure_ascii=False, indent=2)

        with open("./processed_data/label_mapping.json", "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=4)

        train_df.to_csv("./processed_data/train.csv", index=False)
        val_df.to_csv("./processed_data/val.csv", index=False)
        test_df.to_csv("./processed_data/test.csv", index=False)

        mlflow.log_metric("training_set_rows", len(X_train_np))
        mlflow.log_metric("Validate_set_rows", len(X_val_np))
        mlflow.log_metric("test_set_rows", len(X_test_np))

        mlflow.log_artifacts(processed_data_dir, artifact_path="processed_data")
        print("Logged processed data as artifacts in MLflow.")
        
        print("-" * 50)
        print(f"Data preprocessing run finished. Please use the following Run ID for the next step:")
        print(f"Preprocessing Run ID: {run_id}")
        print("-" * 50)
        
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                print(f"run_id={run_id}", file=f)

if __name__ == "__main__":
    preprocess_data()
