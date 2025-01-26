from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

"""
This code includes some auxiliary functions that have been put here in order to be used in different files later

Here are the custom datasets to correctly load the images.
"""

class ClassificationImageDataset(Dataset):
    def __init__(self, root_dir, file_paths, classification_type, transform=None, train=True, validation_split=0.1, random_seed=42):
        """
        This is a Custom Dataset class which inherits Dataset from torch.
        It can be operated in two ways: train=True or train=False
        When train=True the value of validation_split is important but it can be set to None
        root_dir is the directory which contains the files
        file_paths is the path to the file that contains the paths for either training or testing
        """
        super(ClassificationImageDataset).__init__()

        # define useful attributes

        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.classification_type=classification_type

        # Set the label position based on the classification type, used later in extracting the label from the path

        if self.classification_type == 'make':
            label_position = 0
        elif self.classification_type == 'model':
            label_position = 1
        elif self.classification_type == 'part':
            label_position = 3
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
    
class AttributePredictionDataset(Dataset):
    def __init__(self, root_dir, file_paths, prediction_type, transform=None, train=True, validation_split=0.1, random_seed=42):
        """
        This is a Custom Dataset class which inherits Dataset from torch.
        It can be operated in two ways: train=True or train=False
        When train=True the value of validation_split is important but it can be set to None
        root_dir is the directory which contains the files
        file_paths is the path to the file that contains the paths for either training or testing
        """
        super(AttributePredictionDataset).__init__()

        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.prediction_type=prediction_type
        
        # Redefine prediction type in a way that it matches the column names in the attributes.txt file

        if prediction_type=='maxspeed':
            prediction_type='maximum_speed'
        elif prediction_type=='doornumber':
            prediction_type='door_number'
        elif prediction_type=='seatnumber':
            prediction_type='seat_number'
            
        # Import labels but exculude those that don't have the values for the attribe we're predicting

        path=os.path.join(os.getcwd(),'../CompCars/data/misc/attributes.txt')
        label_table=pd.read_csv(path,sep=' ',index_col='model_id')
        label_table=label_table[(label_table[prediction_type]!=0)]

        # Load paths and labels from the text file

        with open(file_paths, 'r') as f:
            for line in f:
                path = line.strip()
                model_id = int(path.split('/')[1])      # Get model id
                
                if model_id in label_table.index:
                    label = label_table.loc[model_id][prediction_type]
                    
                    if prediction_type in ['door_number','seat_number'] and label > 5:  # As is done in the paper, values over 5 are reduced to 5
                        label=5
                        
                    self.image_paths.append(path)   # Save image path
                    self.labels.append(label)  # Since we're using the numbers of either car_make or car_model because the number-label correspondance is wrong

        # If we deal with discrete labels, encode them

        if prediction_type in ['door_number','seat_number','type']:
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
    
class VerificationImageDataset(Dataset):
    def __init__(self, root_dir, file_paths, classification_type, transform=None, train=True, validation_split=0.1, random_seed=42):
        """
        This is a Custom Dataset class which inherits Dataset from torch.
        It can be operated in two ways: train=True or train=False
        When train=True the value of validation_split is important but it can be set to None
        root_dir is the directory which contains the files
        file_paths is the path to the file that contains the paths for either training or testing
        """
        super(VerificationImageDataset).__init__()

        # Define useful attributes

        self.root_dir = root_dir
        self.image1_paths = []
        self.image2_paths = []
        self.labels = []
        self.verification_type=classification_type
        self.train=train

        # Load paths and labels from the text file

        with open(file_paths, 'r') as f:
            for line in f:
                image1,image2,label = line.split(' ')
                self.image1_paths.append(image1)   # Save image1 path
                self.image2_paths.append(image2)   # Save image2 path
                self.labels.append(int(label))  # Save label

        # If train flag is true, split into train and validation sets

        if train:
            train_indices, val_indices = train_test_split(
                range(len(self.image1_paths)),
                test_size=validation_split,
                random_state=random_seed,
                stratify=self.labels  # Ensure good split label-wise
            )
            self.train_indices = train_indices  # Set the indices for later use
            self.val_indices = val_indices

        self.transform = transform

    # Implementing needed methods for Dataset class

    def __len__(self):  
        return len(self.image1_paths)

    def __getitem__(self, idx):
        # Load image and label
        image1_path = os.path.join(self.root_dir, self.image1_paths[idx])
        image2_path = os.path.join(self.root_dir, self.image2_paths[idx])
        image1 = Image.open(image1_path).convert("RGB")  # Ensure 3-channel RGB format
        image2 = Image.open(image2_path).convert("RGB")  # Ensure 3-channel RGB format
        label = self.labels[idx]
        
        # Splits for testing are made for model validation, so we need to fix the labels

        if ((not self.train) and self.verification_type=='make'):
            label1=self.image1_paths[idx].split('/')[0]
            label2=self.image2_paths[idx].split('/')[0]
            
            label=(label1==label2)

        # Apply transformations if any

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, label

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
        
# Define Contrastive Loss which is not present in torch, just an attempt, since it didn't work

class ContrastiveLoss(nn.Module):
    
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, images1, images2, labels):

        # Calculate the Euclidean distance between the image embeddings

        euclidean_distance = F.pairwise_distance(images1, images2, p=2)

        # Compute the contrastive loss

        loss = 0.5 * (labels.float() * torch.pow(euclidean_distance, 2) + 
                      (1 - labels).float() * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        
        # Return the mean loss

        return loss.mean()

# Function used for contrastive loss, useless for now

def contrastive_accuracy(embedding1, embedding2, margin):
    """dist < m --> 1 else 0"""
    euclidean_distance = F.pairwise_distance(embedding1, embedding2, p=2)
    return torch.reshape(euclidean_distance<margin,(1,euclidean_distance.size(0)))
