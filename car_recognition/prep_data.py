from PIL import Image
import os
from glob import glob
import numpy as np
import pandas as pd


class PrepData:
    def __init__(self):
        self.TRAIN_PATH = 'data/training_images/'
        self.TEST_PATH = 'data/testing_images/'
        self.LABELS_PATH = 'data/train_solution_bounding_boxes.csv'
        self.labels = pd.read_csv(self.LABELS_PATH)
        
    def load_images(self):
        img_list_train = list(map(Image.open, glob(f'{self.TRAIN_PATH}*.jpg')))
        img_list_test = list(map(Image.open, glob(f'{self.TEST_PATH}*.jpg')))
        return img_list_train, img_list_test
    
    def preprocess_images(self, img_list):
        formatted_imgs = [np.array(img, dtype='float32') / 255.0 for img in img_list] # Normalize pixel values
        return np.stack(formatted_imgs)
    
    def connect_labels(self, img_list, grid_size, num_boxes, num_classes):
        all_labels = []
        
        for img in img_list:
            # Initialize a 3D array with shape (grid_size, grid_size, num_boxes * 5 + num_classes)
            label_array = np.zeros((grid_size, grid_size, num_boxes * 5 + num_classes))
            
            file_name = os.path.basename(img.filename)
            bounding_boxes = self.labels[self.labels['image'] == file_name]
            
            for index, bounding_box in bounding_boxes.iterrows():
                # Normalize bounding box coordinates
                xmin = bounding_box['xmin'] / img.width
                ymin = bounding_box['ymin'] / img.height
                xmax = bounding_box['xmax'] / img.width
                ymax = bounding_box['ymax'] / img.height
                
                # Convert (xmin, ymin, xmax, ymax) to (x, y, w, h)
                x = (xmin + xmax) / 2
                y = (ymin + ymax) / 2 
                w = xmax - xmin
                h = ymax - ymin
                
                # Determine which grid cell this bounding box falls into
                grid_x = int(x * grid_size)
                grid_y = int(y * grid_size)
                
                # Set the label for this bounding box
                label_array[grid_y, grid_x, :5] = [x * grid_size - grid_x, y * grid_size - grid_y, w, h, 1]  # Normalized x, y to cell, w, h, confidence
                label_array[grid_y, grid_x, 5:] = [1]  # Class probability for the object
                
            all_labels.append(label_array)
    
        return np.array(all_labels)

if __name__ == '__main__':
    p = PrepData()
    train, test = p.load_images()
    labels = p.connect_labels(train)
    print(labels)
    # train, test = p.preprocess_images(train), p.preprocess_images(test)
    