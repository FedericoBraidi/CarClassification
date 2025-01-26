import custom_models as cst
import auxiliaries as aux
import os 
import torch
import torch.nn as nn
from torchvision import transforms
import os
from torch.utils.data import DataLoader, Subset
import gc
from tqdm import tqdm
from torchinfo import summary

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

# Define parameters
prediction_type = 'displacement'    # Choose between 'maxspeed', 'displacement', 'doornumber', 'seatnumber', 'type'
num_epochs = 10
batch_size = 32
learning_rate = 1e-4
splits_folder='train_test_split_full_make_100_80_20_0'  # Folder which contains two files train.txt and test.txt with a list of images to use for train and test
model_name='finetuned-resnet18'  # Choose between 'inceptionv1', 'inceptionmodified', 'resnet18', 'resnet34', 'resnet-simple', 'finetuned-resnet18', 'finetuned-inceptionv1'
patience = 2    # For early stopping
progressive = 1 # Used for differentiating between runs with same parameters
use_data_augmentation=True

# Name the file where we save the model with the parameters used for better readability

model_save_name=f'model_{model_name}_{prediction_type}_{batch_size}_{progressive}.pt'

# Define root path (where the images are) and train path (where the list of images to use is)

root_dir = os.path.join(os.getcwd(), '../CompCars/data/cropped_image')
file_paths_train = os.path.join(os.getcwd(), f'../CompCars/data/splits/{splits_folder}/classification/train.txt')

# Set number of classes

if prediction_type in ['maxspeed', 'displacement']:
    num_classes = 1
elif prediction_type in ['doornumber','seatnumber']:
    num_classes = 4
elif prediction_type == 'type':
    num_classes = 12
else:
    print('Wrong classification type') 

# Create model

if model_name=='inceptionv1':
    model = cst.InceptionV1(in_channels=3, num_classes=num_classes).to(device)
elif model_name=='inceptionmodified':
    model = cst.InceptionModified(in_channels=3, num_classes=num_classes).to(device)
elif model_name=='resnet18':
    model = cst.ResNet(cst.ResidualBlock, [2, 2, 2, 2],num_classes=num_classes).to(device)
elif model_name=='resnet-simple':
    model = cst.ResNet(cst.ResidualBlock, [1, 1, 1, 1],num_classes=num_classes).to(device)
elif model_name=='resnet34':
    model = cst.ResNet(cst.ResidualBlock, [3, 4, 6, 3],num_classes=num_classes).to(device)
elif model_name=='finetuned-resnet18':
    model = cst.FinetuneResnet18(num_classes=num_classes).to(device)
elif model_name=='finetuned-inceptionv1':
    model = cst.FinetuneInceptionV1(num_classes=num_classes).to(device)
else:
    print('Unsupported model')
    
summary(model,(1,3,224,224))

# Define loss and optimizer based on the prediction type

if prediction_type in ['maxspeed','displacement']:    
    criterion = nn.MSELoss(reduction='sum')
elif prediction_type in ['doornumber','seatnumber','type']:
    criterion = nn.CrossEntropyLoss()
else:
    print('Unsupported loss')
    
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0.001)

# Define transformations (resize was arbitrary, normalize was requested from pytorch)

if use_data_augmentation==False:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
else:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Create custom dataset

dataset = aux.AttributePredictionDataset(root_dir=root_dir, file_paths=file_paths_train,
                                         prediction_type=prediction_type,
                                         transform=transform, train=True, validation_split=0.25)

# Create training and validation subsets using the indices calculated during dataset._init_()

train_subset = Subset(dataset, dataset.train_indices)
val_subset = Subset(dataset, dataset.val_indices)

# Create dataloaders for training and validation

train_loader = DataLoader(train_subset, batch_size=batch_size)
valid_loader = DataLoader(val_subset, batch_size=batch_size)

best_val_loss = float('inf')  # For regression tasks
best_val_acc = 0  # For classification tasks
epochs_without_improvement = 0  # Counter for epochs without improvement (early stopping)

# Training loop

for epoch in tqdm(range(num_epochs),leave=False):

    model.train()  # Set model to training mode
    running_loss = 0.0  # Set variables for evaluation
    correct_train = 0
    total_train = 0
    i = 0

    # Loads one batch at a time

    for images, labels in tqdm(train_loader, desc=f'Currently running epoch number {epoch+1}', leave=False):  

        # Move tensors to the configured device

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass

        outputs = model(images)
        
        # Set labels and outputs in the correct format

        if prediction_type in ['maxspeed','displacement']:
            labels = labels.float()     # this is because for MSE loss he wants the labels in float32
            
        loss = criterion(outputs, labels)
        
        i += 1  # Counting batches
        running_loss += loss.item()  # Update running loss

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if prediction_type in ['doornumber','seatnumber','type']:
            # Calculate accuracy
            predicted = torch.argmax(outputs.data, 1)
            total_train += outputs.size(0)
            correct_train += (predicted == labels).sum().item()

        del images, labels, outputs
        torch.cuda.empty_cache()
        gc.collect()

    # Validation loop, it doesn't influence training, it's just to keep track of the model overfitting or not

    model.eval()  # Set model to evaluation mode
    running_val_loss = 0.0  # Create variables for evaluation
    correct_val = 0
    total_val = 0
    val_accuracy = 0

    with torch.no_grad():   # Don't modify model during validation

        for images, labels in tqdm(valid_loader, leave=False):

            # Send tensors to device

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)  # Calculate loss for validation
            running_val_loss += loss.item()  # Update running validation loss

            if prediction_type in ['doornumber','seatnumber','type']:
                
                # Count correctly classified inputs

                predicted = torch.argmax(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

            del images, labels, outputs
            
    
    # Print training and validation stats

    if prediction_type in ['doornumber','seatnumber','type']:

        # Calculate validation accuracy

        val_accuracy = 100 * correct_val / total_val
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss / i}, '
                f'Validation Loss: {running_val_loss / len(valid_loader)}')
        print(f'Training Accuracy: {100 * correct_train / total_train}%, '
                    f'Validation Accuracy: {val_accuracy}%')
    elif prediction_type in ['maxspeed','displacement']:

        # Print RMSE for regression tasks

        print(f'Epoch [{epoch+1}/{num_epochs}], Training Average Difference:{(running_loss/(len(train_loader)*batch_size**2))**(1/2)}, '
                f'Validation Average Difference:{(running_val_loss/(len(valid_loader)*batch_size**2))**(1/2)}')

    # Early stopping logic

    if prediction_type in ['maxspeed', 'displacement']:

        # For regression tasks, use validation loss

        if running_val_loss < best_val_loss:
            best_val_loss = running_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(os.getcwd(), model_save_name))
        else:
            epochs_without_improvement += 1
    elif prediction_type in ['doornumber','seatnumber','type']:

        # For classification tasks, use validation accuracy

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(os.getcwd(), model_save_name))
        else:
            epochs_without_improvement += 1

    # Check for early stopping

    if epochs_without_improvement >= patience:
        print(f'Early stopping triggered after {patience} epochs without improvement in validation accuracy or loss.')
        break

# Save the final model

torch.save(model.state_dict(), os.path.join(os.getcwd(),'../Models', model_save_name))