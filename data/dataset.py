import os
from PIL import Image
import torch
from torch.utils.data import Dataset

from torchvision import transforms

class Shoe40kTransforms:
    def __init__(self, phase):
        if phase == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:  # val or test
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __call__(self, img):
        return self.transform(img)

class Shoe40kDataset(Dataset):
    def __init__(self, df, path, phase):
        self.df = df
        self.path = path
        self.file_names = df['file_name'].values
        self.labels = df['Label'].values
        self.phase = phase
        self.transform = Shoe40kTransforms(phase=phase)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = os.path.join(self.path, file_name)
        
        with Image.open(file_path) as image:
            image = image.convert('RGB')
        
        image = self.transform(image)     
        label = torch.tensor(self.labels[idx]).long()

        return {"pixel_values": image, "label": label}


