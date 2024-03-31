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
    
    def connect_labels(self, img_list, grid_size, anchor_sizes, iou):
        all_labels = []
        
        for img_index, img in enumerate(img_list):
            label_array = np.zeros((grid_size, grid_size, len(anchor_sizes) * 5))
            
            file_name = os.path.basename(img.filename)
            bounding_boxes = self.labels[self.labels['image'] == file_name]
            
            for index, bounding_box in bounding_boxes.iterrows():
                # Normalize bounding box coordinates
                xmin = bounding_box['xmin'] / img.width
                ymin = bounding_box['ymin'] / img.height
                xmax = bounding_box['xmax'] / img.width
                ymax = bounding_box['ymax'] / img.height
                
                # Convert (xmin, ymin, xmax, ymax) to (x, y, w, h)
                x = (xmin + xmax) / 2 # Center x
                y = (ymin + ymax) / 2 # Center y
                w = xmax - xmin
                h = ymax - ymin
                
                # Determine which grid cell this bounding box falls into
                grid_x = int(x * grid_size)
                grid_y = int(y * grid_size)
                
                # Determine which anchor box of the cell has the highest IoU with the bounding box
                best_iou = 0
                best_anchor_index = -1
                for anchor_index, (anchor_w, anchor_h) in enumerate(anchor_sizes):
                    # Calculate IoU with each anchor box
                    iou_score = iou([0, 0, w, h], [0, 0, anchor_w, anchor_h])
                    if iou_score > best_iou:
                        best_iou = iou_score
                        best_anchor_index = anchor_index
                
                if best_anchor_index != -1:  # Valid match found
                    label_index = best_anchor_index * 5
                    # Directly use normalized values; assume predictions will adjust based on anchor
                    label_array[grid_y, grid_x, label_index:label_index+5] = [x, y, w, h, 1] 
                
            all_labels.append(label_array)
    
        return np.array(all_labels)

if __name__ == '__main__':
    p = PrepData()
    train, test = p.load_images()
    labels = p.connect_labels(train, 13, 5, 1)
    print(labels.shape)
    print(labels[0][0][0])
    # train, test = p.preprocess_images(train), p.preprocess_images(test)
    