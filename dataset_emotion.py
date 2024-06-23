import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import BertTokenizer

def load_and_preprocess_emotion_data():
    emotion_df = pd.read_csv("data/tweet_emotions.csv")
    emotion_texts = emotion_df['content'].values
    emotion_labels = emotion_df['sentiment'].values

    # Label encoding
    label_encoder = LabelEncoder()
    emotion_labels = label_encoder.fit_transform(emotion_labels)

    emotion_train_texts, emotion_val_texts, emotion_train_labels, emotion_val_labels = train_test_split(
        emotion_texts, emotion_labels, test_size=0.2, random_state=42)

    # Text preprocessing
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
