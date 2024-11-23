from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This code includes some auxiliary functions that have been put here in order to be used in different files later
"""

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, file_paths, classification_type, transform=None, train=True, validation_split=0.1, random_seed=42):
        """
        This is a Custom Dataset class which inherits Dataset from torch.
        It can be operated in two ways: train=True or train=False
        When train=True the value of validation_split is important but it can be set to None
        root_dir is the directory which contains the files
        file_paths is the path to the file that contains the paths for either training or testing
        """
        super(CustomImageDataset).__init__()

        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.classification_type=classification_type

        if self.classification_type == 'make':
            label_position = 0
        elif self.classification_type == 'model':
            label_position = 1
        else:
            print('Wrong classification type')

        # Load paths and labels from the text file
        with open(file_paths, 'r') as f:
            for line in f:
                path = line.strip()
                label = path.split('/')[label_position]      # Get labels
                self.image_paths.append(path)   # Save image path
                self.labels.append(int(label))  # Since we're using the numbers of either car_make or car_model because the number-label correspondance is wrong
                
        self.label_encoder = LabelEncoder()     # This is needed because the model ids are not sequential and this shifts them as sequential
        self.labels = self.label_encoder.fit_transform(self.labels)

        # If train flag is true, split into train and validation sets
        if train:
            train_indices, val_indices = train_test_split(
                range(len(self.image_paths)),
                test_size=validation_split,
                random_state=random_seed,
                stratify=self.labels  # Ensure good split label-wise
            )
            self.train_indices = train_indices  # Set the indices for later use
            self.val_indices = val_indices

        self.transform = transform

    # Implementing needed methods for Dataset class
    def __len__(self):  
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and label
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_path).convert("RGB")  # Ensure 3-channel RGB format
        label = self.labels[idx]

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, label

# Define Focal Loss which is not present in torch 
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Multiplicative factor
        self.gamma = gamma  # Exponential factor
        self.reduction = reduction  # Can use mean,sum or none

    # Needed function for Loss
    def forward(self, inputs, targets):
        # Apply softmax to get probabilities, as in cross_entropy softmax is computed inside
        probs = F.softmax(inputs, dim=1)
        # Get the probabilities of the true classes
        p_t = probs.gather(1, targets.view(-1, 1))  # Get the predicted probability for the true class
        # Compute the focal loss
        loss = -self.alpha * (1 - p_t) ** self.gamma * torch.log(p_t)
        
        if self.reduction == 'mean':
            return loss.mean()  # Return the mean loss across all samples
        elif self.reduction == 'sum':
            return loss.sum()  # Return the sum of losses
        else:
            return loss  # Return the loss per sample (no reduction)

