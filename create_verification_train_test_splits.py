import os
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm

"""
This code creates the train and test split for the verification task
"""

# Paths and split percentages

percentage_to_test=0.2        # Since the couples given by the authors of the paper are used for testing, we use this to get the number of training samples to get
verification_type='part'      # Verification type, can be 'make', 'model' or 'part'
np.random.seed(0)           # Can be commented out
image_type='part'           # Image type, can be 'full' or 'part'

# Create the filename

filename=f'train_test_split_verification_{verification_type}_{image_type}_{int((1-percentage_to_test)*100)}'

root_path = os.path.join(os.getcwd(),
                          f'../CompCars/data/{'cropped_image' if image_type=='full' else 'part'}')     # Root path of where the images are
write_path = os.path.join(os.getcwd(),
                           f'../CompCars/data/splits/{filename}/verification')     # Path of where to write the txt files with the train test splits

# Define position of label in path

if verification_type=='make':
    id_label=0
elif verification_type=='model':
    id_label=1
elif verification_type=='part':
    id_label=3

# Function to extract labels from paths, set the index to return as 0 for make label, 1 for model label, 3 for part label

def extract_class_label(file_path,id_label):
    return file_path.split('/')[id_label]

# Extract the path to all the files in the root_path tree

tot_filenames=[]

for dirpath, dirnames, filenames in os.walk(root_path):
    for file in filenames:
        rel_path = os.path.join(dirpath, file).split(root_path + '/')[1]    # Eliminate the whole part before the end of root_path
        tot_filenames.append(rel_path)

# Calculate the number of training pairs, they are created so that percentage to testis 20000 pairs, this is done because the test files provided in the article are 20000 couples each

num_to_train = int((20000/(percentage_to_test)) - 20000)
pairs=set()     # This is a set since every pair should only appear at most once in the training set and using a set data structure we have constant look up time  

for i in tqdm(range(num_to_train//2),desc='Creating different pairs',leave=False):            # This is for the different pairs
    
    i1=random.randint(0,len(tot_filenames)-1)     # Extract at random one image
    i2=i1

    while extract_class_label(tot_filenames[i1],id_label)==extract_class_label(tot_filenames[i2],id_label):   # Until the second extracted image is the same label as the first, keep extracting
        
        i2=random.randint(0,len(tot_filenames)-1)
    
    pairs.add((tot_filenames[i1],tot_filenames[i2],0))      # When we extract a second image which is a different label from the first, add it to pairs with 'different' label (0)

for i in tqdm(range(num_to_train//2),desc='Creating equal pairs',leave=False):

    i1=random.randint(0,len(tot_filenames)-1)
    samelabelpath=os.path.join(root_path,os.path.join(*tot_filenames[i1].split('/')[:id_label+1]))  # A bit of a convoluted way to extract the path of the images with the same label
    same_label=[]   
    
    for dirpath,_,filenames in os.walk(samelabelpath):  # Collect all images with the same label
        for file in filenames:
            same_label.append(os.path.join(dirpath, file).split(root_path + '/')[1])
    
    if len(same_label)==1:
        continue
    i2=random.randint(0,len(same_label)-1)  # Extract one image from those with the same label
    while tot_filenames[i1]==same_label[i2]:    # If we extracted the same image, then extract another one
        i2=random.randint(0,len(same_label)-1)  
    
    pairs.add((tot_filenames[i1],same_label[i2],1)) # When we have a couple of images with same label but different from one another, add them with label 'same' (1)

# Save train and split

os.makedirs(write_path, exist_ok=True)

with open(os.path.join(write_path, 'train.txt'), 'w') as file:
    for element in tqdm(pairs,desc='Writing file',leave=False):
        file.write(element[0] + ' ' + element[1] + ' ' + str(element[2]) + '\n') # Write the pairs in the txt file

# In this case, we need to create also the test split from scratch

if image_type=='part': 
    
    # Use 20000 couples for testing
    num_to_test = 20000
    pairs=set()     # This is a set since every pair should only appear at most once in the training set and using a set data structure we have constant look up time  

    for i in tqdm(range(num_to_test//2),desc='Creating different pairs',leave=False):            # This is for the different pairs

        i1=random.randint(0,len(tot_filenames)-1)     # Extract at random one image
        i2=i1

        while extract_class_label(tot_filenames[i1],id_label)==extract_class_label(tot_filenames[i2],id_label):   # Until the second extracted image is the same label as the first, keep extracting
            i2=random.randint(0,len(tot_filenames)-1)
        
        pairs.add((tot_filenames[i1],tot_filenames[i2],0))      # When we extract a second image which is a different label from the first, add it to pairs with 'different' label (0)

    for i in tqdm(range(num_to_test//2),desc='Creating equal pairs',leave=False):
        i1=random.randint(0,len(tot_filenames)-1)
        samelabelpath=os.path.join(root_path,os.path.join(*tot_filenames[i1].split('/')[:id_label+1]))  # A bit of a convoluted way to extract the path of the images with the same label
        same_label=[]   
        
        for dirpath,_,filenames in os.walk(samelabelpath):  # Collect all images with the same label
            for file in filenames:
                same_label.append(os.path.join(dirpath, file).split(root_path + '/')[1])
        
        if len(same_label)==1:
            continue
        i2=random.randint(0,len(same_label)-1)  # Extract one image from those with the same label
        while tot_filenames[i1]==same_label[i2]:    # If we extracted the same image, then extract another one
            i2=random.randint(0,len(same_label)-1)  
        
        pairs.add((tot_filenames[i1],same_label[i2],1)) # When we have a couple of images with same label but different from one another, add them with label 'same' (1)

    # Save train and split
    os.makedirs(write_path, exist_ok=True)

    with open(os.path.join(write_path, 'test.txt'), 'w') as file:
        for element in tqdm(pairs,desc='Writing file',leave=False):
            file.write(element[0] + ' ' + element[1] + ' ' + str(element[2]) + '\n') # Write the pairs in the txt file
            
else:
    
    # Copy the test files in the correct folder

    # Check the operating system and use the respective command
    def copy_file(src,dst):
        if os.name == 'nt':  # Windows
            cmd = f'copy "{src}" "{dst}"'
        else:  # Unix/Linux
            cmd = f'cp "{src}" "{dst}"'
        os.system(cmd)
        
    # Copy File

    copy_file(os.path.join(os.getcwd(),'../CompCars/data/splits/train_test_split_original/verification/verification_pairs_easy.txt'),write_path)
    copy_file(os.path.join(os.getcwd(),'../CompCars/data/splits/train_test_split_original/verification/verification_pairs_medium.txt'),write_path)
    copy_file(os.path.join(os.getcwd(),'../CompCars/data/splits/train_test_split_original/verification/verification_pairs_hard.txt'),write_path)
