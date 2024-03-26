import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_anchor_boxes(grid_size, anchor_sizes):
    rows, cols = grid_size
    anchor_boxes = []
    
    # Calculate step sizes for grid
    step_x = 1.0 / cols
    step_y = 1.0 / rows
    
    for size in anchor_sizes:
        width, height = size
        for y in range(rows):
            for x in range(cols):
                # Offset by 0.5 to get the center of the cell since the grid is from 0 to 1
                center_x = (x + 0.5) * step_x 
                center_y = (y + 0.5) * step_y
                
                anchor_boxes.append((center_x, center_y, width, height))
    
    return anchor_boxes


def apply_offsets(anchor_boxes, offsets):
    new_boxes = []
    
    for box, offset in zip(anchor_boxes, offsets):
        center_x, center_y, width, height = box
        offset_x, offset_y, offset_w, offset_h = offset
        
        new_center_x = center_x + offset_x
        new_center_y = center_y + offset_y
        new_width = width + offset_w
        new_height = height + offset_h
        
        new_boxes.append((new_center_x, new_center_y, new_width, new_height))
    
    return new_boxes


def visualize_anchor_boxes(anchor_boxes, image_size=(400, 400)):
    _, ax = plt.subplots(1)
    ax.set_xlim(0, image_size[0])
    ax.set_ylim(0, image_size[1])
    ax.invert_yaxis()  # Invert y axis to match image coordinates
    
    for box in anchor_boxes:
        center_x, center_y, width, height = box
        # Convert from center, size to bottom-left corner, size
        rect = patches.Rectangle((center_x * image_size[0] - width * image_size[0] / 2, 
                                  center_y * image_size[1] - height * image_size[1] / 2), 
                                 width * image_size[0], height * image_size[1], 
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    plt.show()

    
def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    x1_min, x1_max = x1 - w1 / 2, x1 + w1 / 2
    y1_min, y1_max = y1 - h1 / 2, y1 + h1 / 2
    x2_min, x2_max = x2 - w2 / 2, x2 + w2 / 2
    y2_min, y2_max = y2 - h2 / 2, y2 + h2 / 2
    
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    
    intersection = x_overlap * y_overlap
    union = w1 * h1 + w2 * h2 - intersection
    
    return intersection / union
