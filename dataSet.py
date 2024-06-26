import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
import torch
import csv
from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Read annotations from CSV file
        self.annotations = pd.read_csv(os.path.join(root_dir, 'annotations.csv'))
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'images', self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        hair_density = float(self.annotations.iloc[idx, 1])
        
        if self.transform:
            image = self.transform(image)
        
        return image, hair_density

# Define transforms for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=45),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dete_set = CustomDataset("dataSet/")

with open('dataSet/annotations.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)

    with open('dataSet2/annotations.csv', 'a', newline="") as new_file:
        csv_writer = csv.writer(new_file)

        img_num = 34
        for image, score in dete_set:
            image = transform(image)
            csv_writer.writerow([f"image{img_num}.jpg", f"{score}"])
            save_image(image, 'dataSet2/images/image'+str(img_num)+'.jpg')
            img_num += 1