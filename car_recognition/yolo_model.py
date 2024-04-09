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
    Dropout,
    Resizing
)

from keras.applications import MobileNetV2


def yolo_model(input_shape, grid_dim, num_boxes):
    inputs = Input(shape=input_shape)  # Original shape
    x = Resizing(224, 224, interpolation="bilinear")(inputs)  # Resize to 224x224

    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    # Use the output of the base model as input to the rest of your architecture
    x = base_model(x, training=False)
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Output Layer
    x = Flatten()(x)
    x = Dense(2048)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.5)(x)
    x = Dense(grid_dim * grid_dim * num_boxes * 5, activation='sigmoid')(x) # 5 = 1 (confidence score) + 4 (bounding box coordinates) activation
    x = Reshape((grid_dim, grid_dim, num_boxes * 5))(x)
    
    return tf.keras.Model(base_model.input, x)


if __name__ == '__main__':
    model = yolo_model((448, 448, 3), 7, 2)
    model.summary()
