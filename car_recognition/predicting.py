import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import cv2


def offsets_to_coordinates(prediction, grid_dim, anchor_sizes, x_cell, y_cell, anchor_index):
    x, y, w, h = prediction[0], prediction[1], prediction[2], prediction[3]
    
    # Sigmoid and exp transformations to get the box coordinates from normalized cell coordinates
    x = (tf.sigmoid(x) + x_cell) / grid_dim
    y = (tf.sigmoid(y) + y_cell) / grid_dim
    w = anchor_sizes[anchor_index][0] * tf.exp(w)
    h = anchor_sizes[anchor_index][1] * tf.exp(h)
                
    return x, y, w, h


def draw_boxes(image, boxes):
    image = image.astype(np.uint8) # convert to unsigned 8-bit integers
    for box in boxes:
        x, y, w, h, c = box
        x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2 # convert center coordinates to top-left and bottom-right
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Ensure the image is in BGR format
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image


def decode_netout(netout, anchor_sizes, grid_dim):
    boxes = []
    for row in range(grid_dim):
        for col in range(grid_dim):
            for a in range(len(anchor_sizes)):
                offset = a * 5
                x, y, w, h, confidence = netout[row, col, offset:offset+5]
                x, y, w, h = offsets_to_coordinates((x, y, w, h), grid_dim, anchor_sizes, col, row, a)
                
                boxes.append((x, y, w, h, confidence))
    return boxes


def predict_image(image, model, anchor_sizes, grid_dim):
    image = np.expand_dims(image, axis=0) # add batch dimension
    prediction = model.predict(image)[0] # get the first image in the batch (the only one)
    boxes = decode_netout(prediction, anchor_sizes, grid_dim) 
    return draw_boxes(image[0], boxes)


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
        predictions = [pred for pred in predictions if iou_vectorized(max_confidence[:4], pred[:4]) < iou_threshold]

    return confident_predictions
