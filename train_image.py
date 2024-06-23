import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='transformers.models.bert.modeling_bert')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataset_image import load_and_preprocess_image_data
from models.model import ImageTaggingModel

def train_image_model():
    safebooru_loader = load_and_preprocess_image_data(batch_size=32, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = 10
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
    max_steps_per_epoch = 500 // 32  # 배치 크기를 32로 키우고 최대 스텝 수를 500으로 줄임
    
    print("train start")

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
    train_image_model()
