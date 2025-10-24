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

# โหลดสิ่งที่จำเป็น
try:
    with open("./model.pkl", "rb") as f:
        model = pickle.load(f)

    with open('./processed_data/vocab.json', 'r', encoding='utf-8') as f:
        voc = json.load(f)['idx2word']


    with open('./processed_data/label_mapping.json', 'r', encoding='utf-8') as f:
        act_class = json.load(f)

except FileNotFoundError as E:
    print("Error: ไม่พบไฟล์สักอันหนึ่ง..",E)

# ฟังก์ชันสำหรับแปลงข้อความเป็นตัวเลข (encoding) และปรับความยาวให้คงที่ (padding)
def encode_and_pad(text, max_len=100):
    # แปลงพจนานุกรม voc ที่เก็บ mapping ของคำ -> index ให้กลายเป็น index -> คำ
    # โดยสลับ key, value และแปลง key ให้เป็น int (เพื่อให้แน่ใจว่า index เป็นตัวเลข)
    word2idx = {v: int(k) for k, v in voc.items()}

    # แยกข้อความเป็นคำ (tokens) แล้วแปลงแต่ละคำเป็น index ตาม word2idx
    # ถ้าไม่มีคำในพจนานุกรม (voc) จะใช้ index ของคำว่า "UNK" (unknown) แทน
    # str(...) เพื่อให้ทุก index เป็น string ก่อนจะนำไป join
    tokens = [str(word2idx.get(w, word2idx.get("UNK", 0))) for w in text.split()]

    # ถ้าจำนวน token น้อยกว่า max_len ให้เติม "0" ด้านท้ายจนได้ความยาวครบ 100
    if len(tokens) < max_len:
        tokens += ["0"] * (max_len - len(tokens)) 
    # ถ้ามี token เกินกว่า max_len ให้ตัดส่วนเกินออก
    else:
        tokens = tokens[:max_len]

    # รวม token กลับเป็น string เดียว โดยคั่นด้วยช่องว่าง เช่น "12 45 0 0 0 ..."
    return " ".join(tokens)

# ฟังก์ชันสำหรับทำความสะอาดข้อความก่อนเข้าระบบ (Preprocessing)
def preprocess_text(text):

    # แปลงข้อความทั้งหมดเป็นตัวพิมพ์เล็ก (lowercase)
    text = text.lower()

    # ลบเครื่องหมายวรรคตอน (punctuation) เช่น . , ! ? ด้วย str.translate()
    # โดย str.maketrans('', '', string.punctuation) จะสร้าง mapping ที่ลบทุกเครื่องหมายออก
    text = text.translate(str.maketrans('', '', string.punctuation))

    # ลบช่องว่างเกิน เช่น หลายช่องให้เหลือเพียงช่องเดียว
    text = ' '.join(text.split())

    # ส่งข้อความที่สะอาดแล้วกลับไป
    return text


# text = "Hello, World!!! 123"

# # ขั้นตอนที่ 1: ทำความสะอาดข้อความ
# clean = preprocess_text(text)
# # ผลลัพธ์: "hello world 123"

# # ขั้นตอนที่ 2: แปลงเป็นตัวเลขและจัดความยาว
# encoded = encode_and_pad(clean)
# # ผลลัพธ์ (ตัวอย่าง): "12 34 56 0 0 0 ... (ครบ 100 ตัว)"

# สร้างแอป Flask
app = Flask(__name__)

# -----------------------------
# สร้าง endpoint /predict (ใช้วิธี POST)
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    # รับข้อมูล JSON ที่ส่งมาจาก client (force=True คือให้บังคับอ่านแม้ header ไม่ระบุ JSON)
    data = request.get_json(force=True)
    
    # ดึงค่าข้อความจาก key "text" (ถ้าไม่มีให้เป็นค่าว่าง)
    text = data.get("text", "")

    # ตรวจสอบว่า input ต้องเป็น string เท่านั้น
    if not isinstance(text, str):
        return jsonify({"error": "Input must be a string"}), 400

    # -----------------------------
    # ตรวจสอบจำนวนตัวอักษร ตัวเลข และสัญลักษณ์
    # -----------------------------
    letters = len(re.findall(r"[A-Za-z]", text))        # นับจำนวนตัวอักษรภาษาอังกฤษ
    numbers = len(re.findall(r"[0-9]", text))           # นับจำนวนตัวเลข
    symbols = len(re.findall(r"[^A-Za-z0-9\s]", text))  # นับจำนวนสัญลักษณ์ที่ไม่ใช่ตัวอักษร/ตัวเลข/ช่องว่าง

    # Rule 1: ตรวจสอบว่าสัญลักษณ์มากเกินไปหรือไม่
    if symbols > (letters + numbers):
        return jsonify({"error": "Too many symbols compared to letters/numbers"}), 400

    # Rule 2: ตรวจสอบว่าตัวเลขมากกว่าตัวอักษรหรือไม่
    if numbers > letters:
        return jsonify({"error": "Too many numbers compared to letters"}), 400

    # -----------------------------
    # ส่วนการประมวลผลและพยากรณ์ผลด้วยโมเดล
    # -----------------------------
    try:
        # ขั้นตอนการเตรียมข้อความก่อนเข้าโมเดล
        encoded_text = encode_and_pad(preprocess_text(text))

        # นำข้อความที่เตรียมแล้วเข้าโมเดลเพื่อทำนายผล
        prediction = model.predict([encoded_text])

        # แปลงผลลัพธ์ให้อยู่ในรูป JSON ที่เข้าใจง่าย
        result = {
            'input': data['text'],                        # ข้อความต้นฉบับ
            'predict': act_class.get(str(prediction[0]))  # ผลลัพธ์จากโมเดล (mapping เป็น label)
        }

        # ส่งผลลัพธ์กลับเป็น JSON
        return jsonify(result)

    # หากเกิดข้อผิดพลาดในการทำงานของโมเดลหรือระบบ
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# -----------------------------
# สร้างหน้าเว็บหลัก (GET)
# -----------------------------
@app.route('/', methods=['GET'])
def index():
    # แสดงข้อความแนะนำเมื่อเข้าผ่าน browser
    return "<h1>Emotion Prediction API</h1><p>Use the /predict endpoint with a POST request.</p>"


# -----------------------------
# เริ่มต้นรัน Flask server
# -----------------------------
if __name__ == '__main__':
    # เปิดให้เข้าถึงได้ทุก IP และรันที่ port 5001
    app.run(host='0.0.0.0', port=5001)
