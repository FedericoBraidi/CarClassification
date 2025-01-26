import os
import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

"""
The authors took a subset of the whole dataset and split it into training and testing
then created a train_test_split folder, with a classification folder inside
and inside that two txt files, one with the list of the paths to the training images,
one with the paths to the test images.

This code is used to mimick this.
"""

# Paths and split percentages

percentage_to_use = 1.0         # Percentage of the whole data to use
percentage_to_test = 0.2        # Percentage of the used data to be used during test only
threshold_percentage = 0.0    # If a class has less than this percentage of the total samples, it is not considered
up_bound=False
np.random.seed(0)       # Can be commented out
label_type='part'       # Can be 'make', 'model' or 'part' indicates which type of classification we want to perform
image_type='part'       # Can be 'full' or 'part' indicates which type of image we want to use

# Create filename

filename=f'train_test_split_{image_type}_{label_type}_{int(percentage_to_use*100)}_{int((1-percentage_to_test)*100)}_{int(percentage_to_test*100)}_{str(threshold_percentage).split('.')[1]}_{up_bound}'

# Paths

root_path = os.path.join(os.getcwd(),
                          f'../CompCars/data/{'cropped_image' if image_type=='full' else 'part'}')     # Root path of where the images are
write_path = os.path.join(os.getcwd(),
                           f'../CompCars/data/splits/{filename}/classification')     # Path of where to write the txt files with the train test splits

# Function to extract labels from paths, set the index to return as 0 for make label, 1 for model label

def extract_class_label(file_path,label_type):
    
    if label_type=='make':
        id_label=0
    elif label_type=='model':
        id_label=1
    elif label_type=='part' and image_type=='part':
        id_label=3
    
    return file_path.split('/')[id_label]

# Walk through the folder tree to extract all labels and make the splits evenly

files_by_class = defaultdict(list)
tot_files=0

for dirpath, dirnames, filenames in os.walk(root_path):
    for file in filenames:
        tot_files+=1
        rel_path = os.path.join(dirpath, file).split(root_path + '/')[1]    # Eliminate the whole part before the end of root_path
        class_label = extract_class_label(rel_path,label_type)     
        files_by_class[class_label].append(rel_path)    # Save the path into the corresponding slot for the label

# Print the total number of samples per class to check

print("Total samples per label:")
for class_label, file_list in files_by_class.items():
    print(f"Label {class_label}: {len(file_list)} samples")
    
# Perform the filtering with the thresholds

filtered_files_by_class={}
value_threshold=threshold_percentage*tot_files
top_value_threshold= 4*value_threshold if up_bound else 10000000
tot_filtered_files=0

for label in files_by_class:
    if len(files_by_class[label])>=value_threshold and len(files_by_class[label])<=top_value_threshold:
        filtered_files_by_class[label]=files_by_class[label]
        tot_filtered_files+=len(files_by_class[label])

# Perform even split by label

train_files = []
test_files = []
for class_label, file_list in filtered_files_by_class.items():

    file_list = np.array(file_list)
    num_files = len(file_list)
    num_to_use = int(percentage_to_use * num_files)
    num_to_test = int(num_to_use * percentage_to_test)
    
    # Check if there are enough photos

    if num_to_use<2 or num_to_test<1:
        continue        # Since later the train will further be divided into train and validation, we need to have at least 2 samples
                        # otherwise the scipy function gives an error
    
    # Shuffle indices and split

    indices = np.arange(num_files)
    np.random.shuffle(indices)
    test_indices = indices[:num_to_test]
    train_indices = indices[num_to_test:num_to_use]
    
    train_files.extend(file_list[train_indices])
    test_files.extend(file_list[test_indices])

# Save train and test splits

os.makedirs(write_path, exist_ok=True)

with open(os.path.join(write_path, 'train.txt'), 'w') as file:
    for path in train_files:
        file.write(path + '\n')

with open(os.path.join(write_path, 'test.txt'), 'w') as file:
    for path in test_files:
        file.write(path + '\n')
        
# Print the number of samples per label in train and test sets to check

train_counts = defaultdict(int)
test_counts = defaultdict(int)

for path in train_files:
    label = extract_class_label(path,label_type=label_type)
    train_counts[label] += 1

for path in test_files:
    label = extract_class_label(path,label_type=label_type)
    test_counts[label] += 1

print("\nSamples per label in training set:")
for label, count in train_counts.items():
    print(f"Label {label}: {count} samples")

print("\nSamples per label in testing set:")
for label, count in test_counts.items():
    print(f"Label {label}: {count} samples")
    
# Print dimensions

print(f"Training files: {len(train_files)}, Testing files: {len(test_files)}")
maxlen=0
for key in filtered_files_by_class:
    filtered_files_by_class[key]=(len(filtered_files_by_class[key])/tot_filtered_files)*100
    if filtered_files_by_class[key]>maxlen:
        maxlen=filtered_files_by_class[key]

# Plot distribution of classes

plt.bar(filtered_files_by_class.keys(), filtered_files_by_class.values(), color='g')
plt.show()

