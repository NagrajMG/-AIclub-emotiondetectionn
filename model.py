from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout, Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

def build_net(img_width, img_height, img_depth, num_classes):
    net = Sequential(name='Tuned_CNN')

    # First Convolutional Block
    net.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            input_shape=(img_width, img_height, img_depth),
            kernel_initializer='glorot_uniform',  # Using Xavier initializer
            padding='same',
            name='conv2d_1'
        )
    )
    net.add(BatchNormalization(name='batchnorm_1'))
    net.add(LeakyReLU(alpha=0.1, name='leakyrelu_1'))  # Using LeakyReLU activation
    net.add(MaxPooling2D(pool_size=(2, 2), name='maxpool2d_1'))
    net.add(Dropout(0.3, name='dropout_1'))  # Adjusted dropout rate

    # Second Convolutional Block
    net.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            kernel_initializer='lecun_normal',  # Different initializer
            padding='same',
            name='conv2d_2'
        )
    )
    net.add(BatchNormalization(name='batchnorm_2'))
    net.add(LeakyReLU(alpha=0.1, name='leakyrelu_2'))
    net.add(MaxPooling2D(pool_size=(2, 2), name='maxpool2d_2'))
    net.add(Dropout(0.4, name='dropout_2'))

    # Flatten and Fully Connected Layer
    net.add(Flatten(name='flatten'))
    net.add(
        Dense(
            64,
            kernel_initializer='he_normal',  # Retaining He initializer for dense layers
            name='dense_1'
        )
    )
    net.add(BatchNormalization(name='batchnorm_3'))
    net.add(LeakyReLU(alpha=0.1, name='leakyrelu_3'))
    net.add(Dropout(0.5, name='dropout_3'))

    # Output Layer
    net.add(
        Dense(
            num_classes,
            activation='softmax',
            name='out_layer'
        )
    )

    # Compile the model with a different optimizer
    
    
    return net

if __name__ == '__main__':
    model = build_net(48, 48, 1, 7)
    model.summary()
