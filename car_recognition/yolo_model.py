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
)


def yolo_model(input_shape, grid_dim, num_boxes):
    inputs = Input(shape=input_shape) 
   
    # Initial convolution layer
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    
    # Second convolution and pooling
    x = Conv2D(192, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    
    # Third set of layers
    x = Conv2D(128, (1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(256, (1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(512, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    
    # Fourth set of layers
    for _ in range(4):
        x = Conv2D(256, (1, 1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(512, (3, 3), padding='same', use_bias=False)(x)
        x = LeakyReLU(alpha=0.1)(x)
        
    x = Conv2D(512, (1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(1024, (3, 3), padding='same', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # Additional convolution layers
    for _ in range(2):
        x = Conv2D(512, (1, 1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(1024, (3, 3), padding='same', use_bias=False)(x)
        x = LeakyReLU(alpha=0.1)(x)
    
    x = Conv2D(1024, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(1024, (3, 3), padding='same', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Prepare for the fully connected layer
    x = Flatten()(x)
    x = Dense(4096)(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Output layers
    x = Dense(grid_dim * grid_dim * num_boxes * 5, activation='sigmoid')(x) # 5 = 1 (confidence score) + 4 (bounding box coordinates) activation
    x = Reshape((grid_dim, grid_dim, num_boxes * 5))(x)
    
    return tf.keras.Model(inputs, x)


if __name__ == '__main__':
    model = yolo_model((448, 448, 3), 7, 2)
    model.summary()
