import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='transformers.models.bert.modeling_bert')

import os
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
from PIL import Image
from torchvision import transforms
from models.model import EmotionClassifier, ImageTaggingModel

# 감성 분석 데이터셋 로드 및 전처리
def load_and_preprocess_emotion_data():
    emotion_df = pd.read_csv("data/tweet_emotions.csv")
    emotion_texts = emotion_df['content'].values
    emotion_labels = emotion_df['sentiment'].values

    # 레이블 인코딩
    label_encoder = LabelEncoder()
    emotion_labels = label_encoder.fit_transform(emotion_labels)

    emotion_train_texts, emotion_val_texts, emotion_train_labels, emotion_val_labels = train_test_split(
        emotion_texts, emotion_labels, test_size=0.2, random_state=42)

    # 텍스트 전처리
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = 160

    class EmotionDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )
            return {
                'text': text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }

    emotion_train_dataset = EmotionDataset(emotion_train_texts, emotion_train_labels, tokenizer, max_len)
    emotion_val_dataset = EmotionDataset(emotion_val_texts, emotion_val_labels, tokenizer, max_len)

    emotion_train_loader = DataLoader(emotion_train_dataset, batch_size=16, shuffle=True)
    emotion_val_loader = DataLoader(emotion_val_dataset, batch_size=16)

    return emotion_train_loader, emotion_val_loader, label_encoder.classes_

# 이미지 태깅 데이터셋 로드 및 전처리
def load_and_preprocess_image_data():
    safebooru_metadata_path = "data/all_data.csv"
    safebooru_images_dir = "data/safebooru/images"

    safebooru_df = pd.read_csv(safebooru_metadata_path)
    safebooru_df['image_path'] = safebooru_df['id'].apply(lambda x: os.path.join(safebooru_images_dir, f"{x}.jpg"))

    def check_image_exists(image_path):
        return os.path.exists(image_path)

    safebooru_df = safebooru_df[safebooru_df['image_path'].apply(check_image_exists)]

    print(f"Loaded {len(safebooru_df)} images from SafeBooru dataset.")  # 디버깅 메시지 추가

    if safebooru_df.empty:
        raise ValueError("SafeBooru dataset is empty after filtering non-existent images.")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    class ImageDataset(Dataset):
        def __init__(self, dataframe, transform):
            self.dataframe = dataframe
            self.transform = transform

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            image_path = self.dataframe.iloc[idx]['image_path']
            tags = self.dataframe.iloc[idx]['tags']
            
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            return image, tags

    safebooru_dataset = ImageDataset(safebooru_df, transform)
    safebooru_loader = DataLoader(safebooru_dataset, batch_size=16, shuffle=True)

    return safebooru_loader

# 감성 분석 모델 학습
def train_emotion_model():
    emotion_train_loader, emotion_val_loader, classes = load_and_preprocess_emotion_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(classes)
    model = EmotionClassifier(n_classes=n_classes)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    def train_epoch(model, data_loader, criterion, optimizer, device, max_steps_per_epoch):
        model = model.train()
        losses = []
        correct_predictions = 0
        step = 0

        for d in data_loader:
            if step >= max_steps_per_epoch:
                break

            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            step += 1

        return correct_predictions.double() / (step * data_loader.batch_size), np.mean(losses)

    def eval_model(model, data_loader, criterion, device):
        model = model.eval()
        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for d in data_loader:
                input_ids = d['input_ids'].to(device)
                attention_mask = d['attention_mask'].to(device)
                labels = d['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                _, preds = torch.max(outputs, dim=1)
                loss = criterion(outputs, labels)

                correct_predictions += torch.sum(preds == labels)
                losses.append(loss.item())

        return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

    num_epochs = 5
    best_accuracy = 0
    max_steps_per_epoch = 1000 // 16  # 배치 크기가 16일 때 최대 스텝 수

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            emotion_train_loader,
            criterion,
            optimizer,
            device,
            max_steps_per_epoch
        )

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(
            model,
            emotion_val_loader,
            criterion,
            device
        )

        print(f'Val loss {val_loss} accuracy {val_acc}')
        print()

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'models/emotion_model_state.bin')
            best_accuracy = val_acc

# 이미지 태깅 모델 학습
def train_image_model():
    safebooru_loader = load_and_preprocess_image_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = 10  # n_classes는 데이터셋에 따라 다를 수 있음
    model = ImageTaggingModel(n_classes=n_classes)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    def train_epoch(model, data_loader, criterion, optimizer, device, max_steps_per_epoch):
        model = model.train()
        losses = []
        correct_predictions = 0
        step = 0

        for d in data_loader:
            if step >= max_steps_per_epoch:
                break

            images = d[0].to(device)
            labels = d[1].to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            step += 1

        return correct_predictions.double() / (step * data_loader.batch_size), np.mean(losses)

    def eval_model(model, data_loader, criterion, device):
        model = model.eval()
        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for d in data_loader:
                images = d[0].to(device)
                labels = d[1].to(device)

                outputs = model(images)
                _, preds = torch.max(outputs, dim=1)
                loss = criterion(outputs, labels)

                correct_predictions += torch.sum(preds == labels)
                losses.append(loss.item())

        return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

    num_epochs = 5
    best_accuracy = 0
    max_steps_per_epoch = 1000 // 16  # 배치 크기가 16일 때 최대 스텝 수

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            safebooru_loader,
            criterion,
            optimizer,
            device,
            max_steps_per_epoch
        )

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(
            model,
            safebooru_loader,
            criterion,
            device
        )

        print(f'Val loss {val_loss} accuracy {val_acc}')
        print()

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'models/image_model_state.bin')
            best_accuracy = val_acc

if __name__ == "__main__":
    # train_emotion_model()
    train_image_model()
