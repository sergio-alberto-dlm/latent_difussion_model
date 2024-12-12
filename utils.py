# tools 
import os
import random
from PIL import Image
from dataclasses import dataclass
from tqdm import tqdm 
import yaml

# pytorch 
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

# data tools 
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

def load_config(path):
    """
    Loads config file.

    Args:
        path (str): path to config file.
    Returns:
        cfg (dict): config dict.

    """
    # load configuration from per scene/dataset cfg.
    with open(path, "r") as f:
        cfg = yaml.full_load(f)

    return cfg

class FlowerDataset(Dataset):
    def load_labels(self, mat_file_path):
        # load the .mat file
        data = loadmat(mat_file_path)
        # access the specific variable containing the labels
        labels = data['labels']  
        return torch.tensor(labels).squeeze()


    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, "jpg", fname) for fname in sorted(os.listdir(os.path.join(root_dir, "jpg"))) if fname.endswith(('.jpg', '.png'))]
        self.labels_path = os.path.join(root_dir, "imagelabels.mat")
        self.labels = self.load_labels(self.labels_path)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label 

def get_data(root_dir : str, resize_to : tuple, batch_size : int, num_workers : int, normalize : tuple):
    mean, std = normalize
    # transform data 
    transform = transforms.Compose([
        transforms.Resize(resize_to),
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean, std),
    ])
    dataset = FlowerDataset(root_dir=root_dir, transform=transform)

    # define sizes for training and validation sets
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation

    # split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

def denormalize(tensors : torch.tensor, mean : float, std : float, color_channels : int):
    """Denormalizes image tensors using mean and std provided
    and clip values between 0 and 1"""

    for c in range(color_channels):
        tensors[:, c, :, :].mul_(std[c]).add_(mean[c])

    return torch.clamp(tensors, min=0.0, max=1.0)

def visualize_samples(loader, normalize, color_channels):
    mean, std = normalize
    plt.figure(figsize=(15, 5))

    for images, y in loader:
        images = denormalize(images, mean=mean, std=std, color_channels=color_channels).squeeze()
        target = y.numpy()

        for i in range(8):
            plt.subplot(2, 4, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images[i].permute(1, 2, 0))
            plt.xlabel(target[i], fontsize=20)

        plt.suptitle("Dataset Samples", fontsize=18)
        plt.subplots_adjust(wspace=0.2, hspace=0.2)
        plt.tight_layout()
        plt.show()
        plt.close()

        break

    return
