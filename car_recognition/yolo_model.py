import tensorflow as tf

from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dense,
    LeakyReLU,
    Flatten,
    Reshape,
    BatchNormalization,
    Dropout
)


def yolo_model(input_shape, num_boxes, grid_dim, num_classes):
    input_layer = Input(shape=input_shape )
    x = input_layer
    
    # Layer 1
    x = Conv2D(16, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Layer 2
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Layer 3
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Layer 4
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Layer 5
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Layer 6
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Layer 7
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Output Layer
    x = Flatten()(x)
    x = Dense(4096)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.5)(x)
    x = Dense(grid_dim * grid_dim * num_boxes * 5 * num_classes, activation='sigmoid')(x) # 5 = 1 (confidence score) + 4 (bounding box coordinates)
    x = Reshape((grid_dim, grid_dim, num_boxes * 5 + num_classes))(x)
    
    return tf.keras.Model(input_layer, x)


def custom_loss(labels, predictions):
    # calculate all iou scores
    # calculate the loss for each anchor box by abs(iou - confidence)
    # sum all the losses using mean squared error
    errors = []
    
    predictions = apply_predictions_to_anchor_boxes(predictions)
    
    for i in range(len(predictions)):
        # get the anchor box with the highest iou
        iou = [p.iou(labels[i], predictions[i][j]) for j in range(len(predictions[i]))]
        
        # calculate the loss for all anchor boxes
        for j in range(len(predictions[i])):
            errors.append(abs(iou[j] - predictions[i][j][4]))
            
    # return the mean squared error
    return tf.reduce_mean(tf.square(errors))

if __name__ == '__main__':
    model = yolo_model()
    model.summary()
