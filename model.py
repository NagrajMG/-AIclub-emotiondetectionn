from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout, Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def build_net(img_width, img_height, img_depth, num_classes):

    net = Sequential(name='Tuned_CNN')

    # First Convolutional Block
    net.add(
        Conv2D(
            filters=32,  # Reduced number of filters
            kernel_size=(3, 3),
            input_shape=(img_width, img_height, img_depth),
            kernel_initializer='he_normal',
            padding='same',
            kernel_regularizer=l2(0.01),  # Added L2 regularization
            name='conv2d_1'
        )
    )
    net.add(BatchNormalization(name='batchnorm_1'))
    net.add(LeakyReLU(alpha=0.1, name='leakyrelu_1'))  # Changed to LeakyReLU
    net.add(MaxPooling2D(pool_size=(2, 2), name='maxpool2d_1'))
    net.add(Dropout(0.25, name='dropout_1'))  # Reduced dropout rate

    # Second Convolutional Block
    net.add(
        Conv2D(
            filters=64,  # Reduced number of filters
            kernel_size=(3, 3),
            kernel_initializer='he_normal',
            padding='same',
            kernel_regularizer=l2(0.01),  # Added L2 regularization
            name='conv2d_2'
        )
    )
    net.add(BatchNormalization(name='batchnorm_2'))
    net.add(LeakyReLU(alpha=0.1, name='leakyrelu_2'))
    net.add(MaxPooling2D(pool_size=(2, 2), name='maxpool2d_2'))
    net.add(Dropout(0.3, name='dropout_2'))  # Adjusted dropout rate

    # Third Convolutional Block
    net.add(
        Conv2D(
            filters=128,  # Reduced filters
            kernel_size=(3, 3),
            kernel_initializer='he_normal',
            padding='same',
            kernel_regularizer=l2(0.01),  # Added L2 regularization
            name='conv2d_3'
        )
    )
    net.add(BatchNormalization(name='batchnorm_3'))
    net.add(LeakyReLU(alpha=0.1, name='leakyrelu_3'))
    net.add(MaxPooling2D(pool_size=(2, 2), name='maxpool2d_3'))
    net.add(Dropout(0.4, name='dropout_3'))  # Adjusted dropout rate

    # Fully Connected Layer
    net.add(Flatten(name='flatten'))
    net.add(
        Dense(
            64,  # Reduced neurons
            kernel_initializer='he_normal',
            kernel_regularizer=l2(0.01),  # Added L2 regularization
            name='dense_1'
        )
    )
    net.add(BatchNormalization(name='batchnorm_4'))
    net.add(LeakyReLU(alpha=0.1, name='leakyrelu_4'))
    net.add(Dropout(0.5, name='dropout_4'))  # Adjusted dropout rate

    # Output Layer
    net.add(
        Dense(
            num_classes,
            activation='softmax',
            name='out_layer'
        )
    )
    
    # Compile the model
    optimizer = Adam(learning_rate=0.0005)  # Adjusted learning rate
    net.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return net

if __name__ == '__main__':
    model = build_net(48, 48, 1, 7)
    model.summary()

   
