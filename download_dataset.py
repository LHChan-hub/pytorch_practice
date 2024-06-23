import os
import subprocess

# Kaggle API를 사용하여 데이터셋 다운로드
datasets = [
    'mylesoneill/tagged-anime-illustrations',
    'pashupatigupta/emotion-detection-from-text',
    'williamscott701/memotion-dataset-7k',
    'alamson/safebooru'
]

for dataset in datasets:
    subprocess.run(['kaggle', 'datasets', 'download', '-d', dataset, '-p', 'data'])

files = [
    'data/tagged-anime-illustrations.zip',
    'data/emotion-detection-from-text.zip',
    'data/memotion-dataset-7k.zip',
    'data/safebooru.zip'
]

for file in files:
    subprocess.run(['powershell', 'Expand-Archive', '-Path', file, '-DestinationPath', 'data', '-Force'])
