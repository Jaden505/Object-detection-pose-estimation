import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf


def apply_nms(predictions, iou_threshold=0.7, confidence_threshold=0.7):
    """Apply non-maximum suppression to the predictions"""
    # Filter predictions by confidence threshold first
    predictions = [p for p in predictions if p[4] >= confidence_threshold]
    
    # Sort predictions based on confidence in descending order
    predictions.sort(key=lambda x: x[4], reverse=True)
    
    confident_predictions = []
    while predictions:
        # Take the prediction with the highest confidence
        max_confidence = predictions.pop(0)
        confident_predictions.append(max_confidence)
        
        # Keep only predictions with IoU less than the threshold
        predictions = [pred for pred in predictions if iou_anchor_label(max_confidence[:4], pred[:4]) < iou_threshold]

    return confident_predictions


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
    
    
def iou_anchor_label(box1, box2):
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


def iou_pred_label(box1, box2):
    x1, y1, w1, h1 = tf.split(box1, 4)
    x2, y2, w2, h2 = tf.split(box2, 4)
    
    x1_min, x1_max = x1 - w1 / 2, x1 + w1 / 2
    y1_min, y1_max = y1 - h1 / 2, y1 + h1 / 2
    x2_min, x2_max = x2 - w2 / 2, x2 + w2 / 2
    y2_min, y2_max = y2 - h2 / 2, y2 + h2 / 2
    
    x_overlap = tf.maximum(0.0, tf.minimum(x1_max, x2_max) - tf.maximum(x1_min, x2_min))
    y_overlap = tf.maximum(0.0, tf.minimum(y1_max, y2_max) - tf.maximum(y1_min, y2_min))
    
    intersection = x_overlap * y_overlap
    union = w1 * h1 + w2 * h2 - intersection
    
    return intersection / union