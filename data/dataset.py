import io
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

from torchvision import transforms

class Shoe45kTransforms:
    def __init__(self, phase):
        if phase == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif phase == 'val': 
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __call__(self, img):
        return self.transform(img)




class Shoe45kDataset(Dataset):
    def __init__(self, dataset, phase: str, label_mapping: dict):
        self.dataset = dataset
        self.transforms = Shoe45kTransforms(phase)
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_bytes = item['image']['bytes']
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = self.transforms(image)
        label = self.label_mapping[item['label']]
        label = torch.tensor(label).long()
        return {"pixel_values": image, "label": label, "file_name": item['file_name']}


class BlipDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_bytes = item["image"]['bytes']
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        item["image"] = image
        encoding = self.processor(images=item["image"], padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = item["caption"]
        return encoding

