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
        else:  # val or test
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __call__(self, img):
        return self.transform(img)

class Shoe45kDataset(Dataset):
    def __init__(self, hf_dataset, phase):
        self.dataset = hf_dataset
        self.transform = Shoe45kTransforms(phase)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_path = item['image']['path']
        label = item['label']

        # Load the image
        image = Image.open(image_path).convert('RGB')

        # Apply transforms
        image = self.transform(image)

        return image, label


class BlipDataset(Dataset):
    def __init__(self, hf_dataset, processor):
        self.dataset = hf_dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_path = item['image']
        caption = item['caption']

        # Load the image
        image = PILImage.open(image_path).convert('RGB')

        # Process the image and caption
        encoding = self.processor(images=image, text=caption, padding="max_length", return_tensors="pt")
        encoding = {k: v.squeeze() for k, v in encoding.items()}  # Remove batch dimension

        return encoding