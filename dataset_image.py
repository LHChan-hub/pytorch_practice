import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
import torch

def load_and_preprocess_image_data(batch_size=32, num_workers=4):
    preprocessed_data_path = 'data/safebooru_loader.pt'
    if os.path.exists(preprocessed_data_path):
        print("Loading preprocessed SafeBooru data from file...")
        safebooru_loader = torch.load(preprocessed_data_path)
        return safebooru_loader

    print("Loading SafeBooru metadata...")
    safebooru_metadata_path = "data/all_data.csv"
    safebooru_images_dir = "data/safebooru/images"

    safebooru_df = pd.read_csv(safebooru_metadata_path)
    safebooru_df['image_path'] = safebooru_df['id'].apply(lambda x: os.path.join(safebooru_images_dir, f"{x}.jpg"))

    def check_image_exists(image_path):
        return os.path.exists(image_path)

    safebooru_df = safebooru_df[safebooru_df['image_path'].apply(check_image_exists)]
    print(f"Loaded {len(safebooru_df)} images from SafeBooru dataset.")

    if safebooru_df.empty:
        raise ValueError("SafeBooru dataset is empty after filtering non-existent images.")

    print("Processing SafeBooru images...")
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

    print("Creating image dataset loader...")
    safebooru_dataset = ImageDataset(safebooru_df, transform)
    safebooru_loader = DataLoader(safebooru_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)  # 병렬 처리를 위한 num_workers 추가
    print("Image dataset loader created.")

    # Save preprocessed data
    torch.save(safebooru_loader, preprocessed_data_path)

    return safebooru_loader
