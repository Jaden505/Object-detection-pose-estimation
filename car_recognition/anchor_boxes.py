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
    # If boxes arent type of tensor, convert them to tensor
    if not isinstance(box1, tf.Tensor):
        box1 = tf.convert_to_tensor(box1)
    
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


def iou_vectorized(box1, box2):
    if not isinstance(box1, tf.Tensor):
        box1 = tf.convert_to_tensor(box1)
        box2 = tf.convert_to_tensor(box2)
        
    # Calculate corners of boxes for the intersection area
    box1_corners = tf.concat([box1[..., :2] - box1[..., 2:] / 2.0, 
                              box1[..., :2] + box1[..., 2:] / 2.0], axis=-1) 
    box2_corners = tf.concat([box2[..., :2] - box2[..., 2:] / 2.0, 
                              box2[..., :2] + box2[..., 2:] / 2.0], axis=-1)

    intersect_mins = tf.maximum(box1_corners[..., :2], box2_corners[..., :2])
    intersect_maxs = tf.minimum(box1_corners[..., 2:], box2_corners[..., 2:])
    intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.0)
    intersection = intersect_wh[..., 0] * intersect_wh[..., 1]

    # Calculate union
    box1_area = box1[..., 2] * box1[..., 3]
    box2_area = box2[..., 2] * box2[..., 3]
    union = box1_area + box2_area - intersection

    # Calculate IOU
    iou = intersection / union
    
    return iou


def prediction_to_box(prediction, grid_dim, anchor_sizes, i, j, k):
    x, y, w, h = prediction[0], prediction[1], prediction[2], prediction[3]
    
    # Sigmoid and exp transformations to get the box coordinates from normalized cell coordinates
    x = (tf.sigmoid(x) + i) / grid_dim
    y = (tf.sigmoid(y) + j) / grid_dim
    w = anchor_sizes[k][0] * tf.exp(w)
    h = anchor_sizes[k][1] * tf.exp(h)
                
    return x, y, w, h