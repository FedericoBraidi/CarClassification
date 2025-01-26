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

splits_folder='train_test_split_full_make_100_80_20_0'    # Name of the folder that contains the txt files for the splits 
model_save_name = 'model_finetuned-resnet18_displacement_32_1.pt'  # Name of the model
model_name,prediction_type,batch_size=model_save_name.split('.')[0].split('_')[1:-1]  # Extract parameters of the model to reconstruct it
batch_size = int(batch_size)    # Convert to int

# Just like in the other files

root_dir = os.path.join(os.getcwd(), f'../CompCars/data/cropped_image')
file_paths_test = os.path.join(os.getcwd(), f'../CompCars/data/splits/{splits_folder}/classification/test.txt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformations

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset and DataLoader creation

dataset = aux.AttributePredictionDataset(root_dir=root_dir, file_paths=file_paths_test,
                                        prediction_type=prediction_type, 
                                        transform=transform, train=False)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Set number of classes

if prediction_type in ['maxspeed', 'displacement']:
    num_classes = 1
elif prediction_type in ['doornumber','seatnumber']:
    num_classes = 4
elif prediction_type == 'type':
    num_classes = 12
else:
    print('Wrong classification type') 

# Create model equal to the one we want to load

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

# Define loss and optimizer

if prediction_type in ['maxspeed','displacement']:    
    criterion = nn.MSELoss(reduction='sum')
elif prediction_type in ['doornumber','seatnumber','type']:
    criterion = nn.CrossEntropyLoss()
else:
    print('Unsupported loss')

# Load model and move to device

model.load_state_dict(torch.load(os.path.join(os.getcwd() ,model_save_name), map_location=device))

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
        
        if prediction_type in ['doornumber','seatnumber','type']:
        
            # Get predictions
        
            _, preds = torch.max(outputs, 1)  # Get the predicted class indices
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

# Calculate metrics

average_loss = total_loss / len(dataset)

if prediction_type in ['doornumber','seatnumber','type']:

    accuracy = accuracy_score(all_labels, all_preds)

# Print results

if prediction_type in ['doornumber','seatnumber','type']:

    print(f"Test Accuracy: {accuracy * 100}%")
    print(f"Test Loss: {average_loss}")

elif prediction_type in ['maxspeed','displacement']:

    print(f"Test Average Difference: {(average_loss/(batch_size**2))**(1/2)}")