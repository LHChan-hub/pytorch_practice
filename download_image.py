import os
import requests
import pandas as pd
from tqdm import tqdm

def download_images(metadata_path, image_dir):
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    metadata = pd.read_csv(metadata_path)

    for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
        image_url = "https:" + row['preview_url']
        image_id = row['id']
        image_path = os.path.join(image_dir, f"{image_id}.jpg")
        
        if os.path.exists(image_path):
            continue

        try:
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                with open(image_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
        except Exception as e:
            print(f"Failed to download {image_url}: {e}")

if __name__ == "__main__":
    metadata_path = "data/all_data.csv"
    image_dir = "data/safebooru/images"
    download_images(metadata_path, image_dir)
