import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import cv2


def labels_to_coordinates(labels, img_width, img_height, grid_size):
    boxes = []  # List to hold boxes in format suitable for drawing (x1, y1, x2, y2)
    for grid_y in range(grid_size):
        for grid_x in range(grid_size):
            for i in range(len(labels[0, 0]) // 5):  # Iterate through each anchor
                label = labels[grid_y, grid_x, i*5:(i+1)*5]
                if label[4] > 0:  # Check if confidence > 0
                    # Extract normalized center coordinates, width, and height
                    cx, cy, w, h, confidence = label

                    # Convert to pixel coordinates
                    x_center_pixel = cx * img_width
                    y_center_pixel = cy * img_height
                    width_pixel = w * img_width
                    height_pixel = h * img_height

                    # Convert to corner coordinates
                    x1, y1 = int(x_center_pixel - width_pixel / 2), int(y_center_pixel - height_pixel / 2)
                    x2, y2 = int(x_center_pixel + width_pixel / 2), int(y_center_pixel + height_pixel / 2)

                    boxes.append((x1, y1, x2, y2, confidence))
    return boxes


def predict_image(image, model, grid_dim, iou_vectorized):
    image = np.expand_dims(image, axis=0) # add batch dimension
    prediction = model.predict(image)[0] # get the first image in the batch (the only one)
    boxes = labels_to_coordinates(prediction, image.shape[2], image.shape[1], grid_dim)
    boxes = apply_nms(boxes, iou_vectorized)
    return draw_boxes(np.squeeze(image[0]), boxes)


def draw_boxes(image, boxes):
    for box in boxes:
        y1, x1, y2, x2 = box[:4]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    return image

def apply_nms(boxes, iou_vectorized, iou_threshold=0.7, confidence_threshold=0.7):
    """ Apply non-maximum suppression to the predictions
        grouping predictions with high Iou that are close to each other.
        Close, high IoU predictions often represent the same object.
    """
    # Filter predictions by confidence threshold first
    boxes = [p for p in boxes if p[4] >= confidence_threshold]

    # Sort predictions based on confidence in descending order
    boxes.sort(key=lambda x: x[4], reverse=True)

    confident_predictions = []
    while boxes:
        # Take the prediction with the highest confidence
        max_confidence = boxes.pop(0)
        confident_predictions.append(max_confidence)

        # Keep only predictions with IoU less than the threshold
        boxes = [pred for pred in boxes if iou_vectorized(max_confidence[:4], pred[:4]) < iou_threshold]

    return confident_predictions
