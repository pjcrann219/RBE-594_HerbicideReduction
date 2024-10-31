import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir, num_images=None):
        """
        Sets up dataset paths and transforms
        """
        self.root_dir = root_dir
        # Resize to 256x256, crop to 224x224, send to tensor, normalize by ResNet Values
        self.rgb_transform = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        # Resize to 256x256, crop to 224x224, send to tensor, not normalizing right now
        self.nir_transform = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        self.image_names = os.listdir(self.root_dir + '/images/nir')

        if num_images is not None:
            self.image_names = self.image_names[:min(num_images, len(self.image_names))]

    def __len__(self):
        """Number of samples in dataset"""
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        Loads RGB and NIR images and corresponding weed cluster label
        Returns combined 4-channel input tensor and binary label
        """
        
        # Grab image name
        image_name = self.image_names[idx]

        # Grab image paths
        rgb_path = self.root_dir + '/images/rgb/' + image_name
        nir_path = self.root_dir + '/images/nir/' + image_name
        output_path = self.root_dir + '/labels/weed_cluster/' + image_name.replace('.jpg', '.png')

        # Convert input images to 1 tensor
        rgb_tensor = self.rgb_transform(Image.open(rgb_path))
        nir_tensor = self.nir_transform(Image.open(nir_path))
        input = torch.cat((rgb_tensor, nir_tensor), dim=0)

        # Convert label to tensor
        output_tensor = transforms.ToTensor()(Image.open(output_path))
        # label = output_tensor.sum() == 0
        label = torch.tensor(int(output_tensor.sum() > 0), dtype=torch.float)
        
        return input, label

def get_dataloader(root_dir, batch_size=4, num_workers=0, shuffle=False, num_images=None):
    """
    Creates dataloader with specified batch size and workers
    """
    dataset = CustomDataset(root_dir, num_images=num_images)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader

class TrainingData:
    def __init__(self):
        self.hyper_params = []
        self.epoch = []
        self.training_loss = []
        self.training_accuracy = []
        self.testing_loss = []
        self.testing_accuracy = []


# # Example usage
# if __name__ == "__main__":
#     root_dir = "ComputerVision/Agriculture-Vision-2021/train"
#     dataloader = get_dataloader(root_dir)
#     for input, labels, in dataloader:
#         print(input.shape)
#         print(labels.shape)
#         break