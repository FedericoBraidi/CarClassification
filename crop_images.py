import os
import cv2
from tqdm import tqdm

"""
This code should ideally only be run once to crop the images
"""

# Root path that contains everything

root_path=os.path.join(os.getcwd(),'../CompCars/data')

# Walk through the image folder

for subdir, dirs, files in tqdm(os.walk(os.path.join(root_path,'image'))):
    for file in files:

        # Build abs path of image and of file containing cropping info about it

        image_path=os.path.join(subdir,file)
        info_path=image_path.replace('image','label').replace('jpg','txt')

        # Open the file with the cropping info, extract the bounding points and crop the image

        with open(info_path,'r') as f:
            x1,y1,x2,y2=[int(val) for val in f.readlines()[2].replace('\n','').split(sep=' ')]  # Get the boundaries
            image = cv2.imread(image_path)  # Read the image
            cropped_image = image[y1:y2, x1:x2, :]  # Crop the image

            make, model, year, filename = image_path[image_path.find('image')+6:].split('/')    # Get folder structure to reproduce it

            cropped_path=os.path.join(root_path, 'cropped_image', make, model, year)    # Create cropped path

            # Save cropped image
            
            if not os.path.exists(cropped_path):
                try:
                    os.makedirs(cropped_path)
                except:
                    pass
            if not os.path.exists(os.path.join(cropped_path,filename)):
                try:
                    cv2.imwrite(os.path.join(cropped_path, filename), cropped_image)
                except:
                    pass

