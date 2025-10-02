import numpy as np
from flask import Flask, request, jsonify
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
import pickle
import json
import string
import re
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

try:
    with open("./model.pkl", "rb") as f:
        model = pickle.load(f)

    with open('./processed_data/vocab.json', 'r', encoding='utf-8') as f:
        voc = json.load(f)['idx2word']


    with open('./processed_data/label_mapping.json', 'r', encoding='utf-8') as f:
        act_class = json.load(f)

except FileNotFoundError:
    print("Error: ไม่พบไฟล์สักอันหนึ่ง..")


def encode_and_pad(text, max_len=100):
    word2idx = {v: int(k) for k, v in voc.items()}

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

app = Flask(__name__)

@app.route('/predict', methods=['POST'])

def predict():
    data = request.get_json(force=True)
    text = data.get("text", "")

    if not isinstance(text, str):
        return jsonify({"error": "Input must be a string"}), 400

    # Count categories
    letters = len(re.findall(r"[A-Za-z]", text))
    numbers = len(re.findall(r"[0-9]", text))
    symbols = len(re.findall(r"[^A-Za-z0-9\s]", text))  # everything else (punctuation, emoji, etc.)

    # Rule 1: Too many symbols
    if symbols > (letters + numbers):
        return jsonify({"error": "Too many symbols compared to letters/numbers"}), 400

    # Rule 2: Too many numbers
    if numbers > letters:
        return jsonify({"error": "Too many numbers compared to letters"}), 400
    try:

        encoded_text = encode_and_pad(preprocess_text(text))

        prediction = model.predict([encoded_text])

        result = {
            'input': data['text'],
            'predict': act_class.get(str(prediction[0]))
        }

        return jsonify(result)


    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Endpoint หลักสำหรับทดสอบว่า API ทำงานหรือไม่
@app.route('/', methods=['GET'])
def index():
    return "<h1>Iris Prediction API</h1><p>Use the /predict endpoint with a POST request.</p>"


# รัน Flask server
if __name__ == '__main__':
    # app.run(debug=True) # ใช้สำหรับตอนพัฒนา
    app.run(host='0.0.0.0', port=5001) # ใช้สำหรับ production หรือให้เครื่องอื่นเรียกได้
