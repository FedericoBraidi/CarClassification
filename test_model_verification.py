import torch
import custom_models as cst
import os
import auxiliaries as aux
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch.nn as nn
from torchinfo import summary

# Parameters
splits_folder='train_test_split_verification_full_model_80'    # Name of the folder that contains the txt files for the splits 
model_save_name = 'model_verification_inceptionmodified_model_32_binary-cross-entropy_1.pt'  # Name of the model
model_name,classification_type,batch_size,loss_name=model_save_name.split('.')[0].split('_')[2:-1]  # Extract parameters of the model to reconstruct it
batch_size = int(batch_size)    # Convert to int
contrastive_margin=16
classification_threshold=0.6

image_type=splits_folder.split('_')[4]

# Just like in the other files
root_dir = os.path.join(os.getcwd(), f'../CompCars/data/{'cropped_image' if image_type=='full' else 'part'}')
file_paths_test = os.path.join(os.getcwd(), f'../CompCars/data/splits/{splits_folder}/verification/test.txt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if loss_name=='contrastive':
    criterion = aux.ContrastiveLoss(margin=contrastive_margin)
elif loss_name=='binary-cross-entropy':
    criterion = nn.BCELoss()

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

model.fc1 = nn.Identity()
    
model = cst.SiameseNetwork(model,contra_loss=True if loss_name=='contrastive' else False).to(device)

summary(model,((1,3,224,224),(1,3,224,224)))
# Load model and move to device
model.load_state_dict(torch.load(os.path.join(os.getcwd(), '../Models',model_save_name), map_location=device))

# Set in evaluation mode
model.eval()

if image_type=='part':

        # Dataset and DataLoader creation
        dataset = aux.VerificationImageDataset(root_dir=root_dir, file_paths=file_paths_test,
                                        classification_type=classification_type, 
                                        transform=transform, train=False)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        running_loss = 0.0  # Set variables for evaluation
        correct_test = 0
        total_test = 0
        i = 0

        # Evaluation
        with torch.no_grad():
            for images1, images2, labels in tqdm(test_loader, desc="Testing"):
                #Move to device
                images1, images2, labels = images1.to(device), images2.to(device), labels.to(device) 

                if loss_name=='binary-cross-entropy':
                    labels = labels.float()
                
                #loss = criterion(similarity.squeeze(), labels)
                
                #print(criterion(outputs,labels))

                if loss_name=='contrastive':
                    output1, output2 = model(images1, images2)
                    loss = criterion(output1, output2, labels)
                    predicted = aux.contrastive_accuracy(output1, output2, margin=contrastive_margin)
                    
                else:
                    output_labels_prob = model(images1, images2)
                    loss = criterion(output_labels_prob.squeeze(), labels)
                    predicted = torch.reshape(output_labels_prob>classification_threshold,(1,output_labels_prob.size(0)))
                    
                running_loss += loss.item()  # Update running loss

                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        # Calculate metrics
        accuracy = correct_test / total_test
        average_loss = running_loss / len(dataset)

        # Print results
        print(f"Test Loss: {average_loss}")
        print(f"Test Accuracy: {accuracy * 100}%")

elif image_type=='full':
    
    test_levels=['easy','medium','hard']
    
    for level in test_levels:
        
        file_paths_test = os.path.join(os.getcwd(), f'../CompCars/data/splits/{splits_folder}/verification/verification_pairs_{level}.txt')
        
        dataset = aux.VerificationImageDataset(root_dir=root_dir, file_paths=file_paths_test,
                                classification_type=classification_type, 
                                transform=transform, train=False)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        running_loss = 0.0  # Set variables for evaluation
        correct_test = 0
        total_test = 0
        i = 0

        # Evaluation
        with torch.no_grad():
            for images1, images2, labels in tqdm(test_loader, desc="Testing"):
                #Move to device
                images1, images2, labels = images1.to(device), images2.to(device), labels.to(device) 

                if loss_name=='binary-cross-entropy':
                    labels = labels.float()
                
                #loss = criterion(similarity.squeeze(), labels)
                
                #print(criterion(outputs,labels))

                if loss_name=='contrastive':
                    output1, output2 = model(images1, images2)
                    loss = criterion(output1, output2, labels)
                    predicted = aux.contrastive_accuracy(output1, output2, margin=contrastive_margin)
                    
                else:
                    output_labels_prob = model(images1, images2)
                    loss = criterion(output_labels_prob.squeeze(), labels)
                    predicted = torch.reshape(output_labels_prob>classification_threshold,(1,output_labels_prob.size(0)))
                    
                running_loss += loss.item()  # Update running loss

                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        # Calculate metrics
        accuracy = correct_test / total_test
        average_loss = running_loss / len(dataset)

        # Print results
        print(f"{level} Test Loss: {average_loss}")
        print(f"{level} Test Accuracy: {accuracy * 100}%")
