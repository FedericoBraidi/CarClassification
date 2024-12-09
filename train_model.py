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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

# Define parameters
classification_type = 'make'  # Make or model
num_epochs = 10
batch_size = 32
learning_rate = 1e-4
splits_folder='train_test_split3'  # Folder which contains two files train.txt and test.txt with a list of images to use for train and test
model_name='inception'  # Model, for now between inception, resnet18 and resnet-simple                     
loss_name='focal'   # Loss, for now between focal and cross-entropy
patience = 2    # For early stopping
progressive = 3 # Used for differentiating between runs with same parameters

# Name the file where we save the model with the parameters used for better readability
model_save_name=f'model_{model_name}_{classification_type}_{batch_size}_{loss_name}_{progressive}.pt'

# Define root path (where the images are) and train path (where the list of images to use is)
root_dir = os.path.join(os.getcwd(), '../CompCars/data/cropped_image')
file_paths_train = os.path.join(os.getcwd(), f'../CompCars/data/{splits_folder}/classification/train.txt')

# Set number of classes
if classification_type == 'make':
    num_classes = 163
elif classification_type == 'model':
    num_classes = 1712
else:
    print('Wrong classification type') 

# Create model
if model_name=='inception':
    model = cst.InceptionV1(in_channels=3, num_classes=num_classes).to(device)
elif model_name=='resnet18':
    model = cst.ResNet(cst.ResidualBlock, [2, 2, 2, 2],num_classes=num_classes).to(device)
elif model_name=='resnet-simple':
    model = cst.ResNet(cst.ResidualBlock, [1, 1, 1, 1],num_classes=num_classes).to(device)
else:
    print('Unsupported model')

# Define loss and optimizer
if loss_name=='focal':    
    criterion = aux.FocalLoss()
elif loss_name=='cross-entropy':
    criterion = nn.CrossEntropyLoss()
else:
    print('Unsupported loss')
    
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0.001)

# Define transformations (resize was arbitrary, normalize was requested from pytorch)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create custom dataset
dataset = aux.CustomImageDataset(root_dir=root_dir, file_paths=file_paths_train, 
                                 classification_type=classification_type, 
                                 transform=transform, train=True, validation_split=0.25)

# Create training and validation subsets using the indices calculated during dataset.__init__()
train_subset = Subset(dataset, dataset.train_indices)
val_subset = Subset(dataset, dataset.val_indices)

# Create dataloaders for training and validation
train_loader = DataLoader(train_subset, batch_size=batch_size)
valid_loader = DataLoader(val_subset, batch_size=batch_size)

best_val_acc = 0  # Initialize best validation accuracy
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
        loss = criterion(outputs, labels)
        i += 1  # Counting batches
        running_loss += loss.item()  # Update running loss

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

    with torch.no_grad():   # Don't modify model during validation
        for images, labels in tqdm(valid_loader, leave=False):
            # Send tensors to device
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)  # Calculate loss for validation
            running_val_loss += loss.item()  # Update running validation loss

            # Count correctly classified inputs
            predicted = torch.argmax(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

            del images, labels, outputs

    # Calculate validation accuracy
    val_accuracy = 100 * correct_val / total_val

    # Print training and validation stats
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss / i}, Training Accuracy: {100 * correct_train / total_train}%, '
          f'Validation Loss: {running_val_loss / len(valid_loader)}, Validation Accuracy: {val_accuracy}%')

    # Early stopping check
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        epochs_without_improvement = 0  # Reset counter if there's improvement
        # Save the best model
        torch.save(model.state_dict(), os.path.join(os.getcwd(), model_save_name))
    else:
        epochs_without_improvement += 1

    # If patience is exceeded, stop training
    if epochs_without_improvement >= patience:
        print(f'Early stopping triggered after {epoch+1} epochs without improvement in validation accuracy.')
        break

# Save the model (final model if no early stopping occurred)
torch.save(model.state_dict(), os.path.join(os.getcwd(), model_save_name))