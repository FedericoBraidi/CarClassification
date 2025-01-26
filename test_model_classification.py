import torch
import custom_models as cst
import os
import auxiliaries as aux
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch.nn as nn

# Parameters

splits_folder='train_test_split_part_model_100_80_20_0'    # Name of the folder that contains the txt files for the splits 
model_save_name = 'model_inceptionmodified_model_32_focal_6.pt'  # Name of the model to be tested
model_name,classification_type,batch_size,loss_name=model_save_name.split('.')[0].split('_')[1:-1]  # Extract parameters of the model to reconstruct it
batch_size = int(batch_size)    # Convert to int

image_type=splits_folder.split('_')[3]

# Just like in the other files

root_dir = os.path.join(os.getcwd(), f'../CompCars/data/{'cropped_image' if image_type=='full' else 'part'}')
file_paths_test = os.path.join(os.getcwd(), f'../CompCars/data/splits/{splits_folder}/classification/test.txt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformations

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset and DataLoader creation

dataset = aux.ClassificationImageDataset(root_dir=root_dir, file_paths=file_paths_test,
                                 classification_type=classification_type, 
                                 transform=transform, train=False)  # Use train=False to avoid the validation split
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Number of classes based on classification type

if classification_type == 'make':
    if image_type=='full':
        num_classes = 163
    elif image_type=='part':
        num_classes = 123
elif classification_type == 'model':
    if image_type == 'full':    
        num_classes = 1716
    elif image_type == 'part':
        num_classes = 956
elif classification_type == 'part':
    num_classes=8
else:
    raise ValueError("Unsupported classification type. Use 'make' or 'model'.")

# Create model equal to the one we want to load

if model_name=='inceptionv1':
    model = cst.InceptionV1(in_channels=3, num_classes=num_classes).to(device)
elif model_name=='inceptionmodified':
    model = cst.InceptionModified(in_channels=3, num_classes=num_classes).to(device)
elif model_name=='resnet18':
    model = cst.ResNet(cst.ResidualBlock, [2, 2, 2, 2],num_classes=num_classes).to(device)
elif model_name=='resnet-simple':
    model = cst.ResNet(cst.ResidualBlock, [1, 1, 1, 1],num_classes=num_classes).to(device)
elif model_name=='finetuned-resnet18':
    model = cst.FinetuneResnet18(num_classes=num_classes).to(device)
elif model_name=='finetuned-inceptionv1':
    model = cst.FinetuneInceptionV1(num_classes=num_classes).to(device)
else:
    print('Unsupported model')

# Loss definition, based on the one used during training

if loss_name=='focal':    
    criterion = aux.FocalLoss()
elif loss_name=='cross-entropy':
    criterion = nn.CrossEntropyLoss()
else:
    print('Unsupported loss')

# Load model and move to device

model.load_state_dict(torch.load(os.path.join(os.getcwd(),'../Models',model_save_name), map_location=device))

# Set in evaluation mode

model.eval()
all_preds = []
all_labels = []
total_loss = 0.0

# Evaluation

with torch.no_grad():

    for images, labels in tqdm(test_loader, desc="Testing"):

        #Move to device

        images, labels = images.to(device), labels.to(device)
        
        # Calculate loss
        
        if model_name=='inceptionv1':
            outputs,_,_ = model(images)
        else:
            outputs = model(images)
        
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)  # Multiply by batch size for total loss 
                                                    #(its not exactly right, the last batch
                                                    # might not be exactly of batch_size)
        
        # Get predictions

        _, preds = torch.max(outputs, 1)  # Get the predicted class indices
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate metrics

accuracy = accuracy_score(all_labels, all_preds)
average_loss = total_loss / len(dataset)

# Print results

print(f"Test Loss: {average_loss}")
print(f"Test Accuracy: {accuracy * 100}%")
