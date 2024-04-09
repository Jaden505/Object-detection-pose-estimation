import tensorflow as tf


def iou_vectorized(box1, box2):
    if not isinstance(box1, tf.Tensor):
        box1 = tf.convert_to_tensor(box1)
    if not isinstance(box2, tf.Tensor):
        box2 = tf.convert_to_tensor(box2)
        
    # make sure both boxes have same type
    box1 = tf.cast(box1, tf.float32)
    box2 = tf.cast(box2, tf.float32)
        
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
    iou = intersection / (union + 1e-7)  # Adding a small epsilon value to avoid division by zero
    
    return iou


def offsets_to_coordinates(prediction, grid_dim, anchor_sizes, x_cell, y_cell, anchor_index):
    x, y, w, h = prediction[0], prediction[1], prediction[2], prediction[3]
    
    # Sigmoid and exp transformations to get the box coordinates from normalized cell coordinates
    x = (tf.sigmoid(x) + x_cell) / grid_dim
    y = (tf.sigmoid(y) + y_cell) / grid_dim
    w = anchor_sizes[anchor_index][0] * tf.exp(w)
    h = anchor_sizes[anchor_index][1] * tf.exp(h)
                
    return x, y, w, h