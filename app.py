import os
import torch
from flask import Flask, request, render_template
from transformers import BertTokenizer
from models.model import EmotionClassifier
from dataset_emotion import load_and_preprocess_emotion_data
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='transformers.models.bert.modeling_bert')

app = Flask(__name__)

print("Loading emotion model...")
# Model loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_model = EmotionClassifier(n_classes=13)  # 저장된 모델의 클래스 수로 변경
emotion_model.load_state_dict(torch.load('models/emotion_model_state.bin', map_location=device))
emotion_model = emotion_model.to(device)
emotion_model.eval()
print("Emotion model loaded.")

# Tokenizer loading
print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("Tokenizer loaded.")

# Load label encoder
print("Loading and processing emotion dataset...")
emotion_train_loader, emotion_val_loader, label_classes = load_and_preprocess_emotion_data()
label_encoder = {idx: label for idx, label in enumerate(label_classes)}
print("Emotion dataset loaded and processed.")
print("Available emotion tags:", label_classes)  # 레이블 클래스 출력

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=160,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        output = emotion_model(input_ids, attention_mask)
        _, prediction = torch.max(output, dim=1)

    prediction = prediction.item()
    tags = label_encoder[prediction]
    
    return render_template('result.html', prediction=tags)

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(host='0.0.0.0', port=5000, debug=True)  # 디버그 모드 활성화
