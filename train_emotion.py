import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='transformers.models.bert.modeling_bert')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np  # numpy 모듈 import 추가
from dataset_emotion import load_and_preprocess_emotion_data
from models.model import EmotionClassifier

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

if __name__ == "__main__":
    train_emotion_model()
