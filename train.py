
#学习率调试，首先我们设置一个较小的学习率 查看loss的变化情况 使用Tensorboard记录下来
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint

from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        TensorBoard)
from tensorflow.keras.optimizers import Adam
from model import create_model
import numpy as np
def train():
    log_dir = '/kaggle/working/logs' #训练日志路劲
    X_train = np.load('/kaggle/input/numpy-array-of-images-and-labels/X_train.npy')  # Replace with your actual path
    y_train = np.load('/kaggle/input/numpy-array-of-images-and-labels/y_train.npy')  # Replace with your actual path
    X_val = np.load('/kaggle/input/numpy-array-of-images-and-labels/X_val.npy')      # Replace with your actual path
    y_val = np.load('/kaggle/input/numpy-array-of-images-and-labels/y_val.npy')      # Replace with your actual path
    X_test = np.load('/kaggle/input/numpy-array-of-images-and-labels/X_test.npy')    # Replace with your actual path
    X_train = X_train.reshape((X_train.shape[0], 48, 48, 1))
    X_val = X_val.reshape((X_val.shape[0], 48, 48, 1))
    X_test = X_test.reshape((X_test.shape[0], 48, 48, 1))
    batch_size = 25
    # 加载数据集
    lr = 1e-3
    epochs = 200
    num_classes=7 #你的分类数
    train_datagen = ImageDataGenerator( #数据集增强，这些参数查阅keras 官方文档 我前面的博客VGG 中 说明过也有介绍说
        # rescale=1 / 255.0,
        rotation_range = 10,
        zoom_range = 0.1,
        horizontal_flip = True
       )

    train_generator = train_datagen.flow(
        X_train, y_train,
        # target_size=(48, 48),
        # color_mode="grayscale",
        batch_size=batch_size,
        # class_mode="categorical",
        shuffle=True,
        seed=42
    )
    test_datagen = ImageDataGenerator(
        rescale=1 / 255.0)
        
    valid_generator = test_datagen.flow(
        X_val, y_val,
        # target_size=(48, 48),
        # color_mode="grayscale",
        # class_mode="categorical",
        batch_size=batch_size,
        shuffle=False,
        seed=42
    )
    test_gen = test_datagen.flow(
        X_test,
        # target_size=(48, 48),
        # color_mode="grayscale",
        batch_size=batch_size,
        shuffle=False,
        seed=42
    )
    #你的模型，模型参数自己调试
    model = create_model(num_classes=num_classes)

    model.summary()
    training_weights='/kaggle/working/weights'  #这里是保存每次训练权重的  如果需要自己取消注释
    checkpoint_period = ModelCheckpoint(training_weights + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                        monitor='val_loss', save_weights_only=True, save_best_only=False, save_freq='epoch', verbose = 1)
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1) #学习率衰减
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1) # val_loss 不下降时 停止训练 防止过拟合
    tensorboard = TensorBoard(log_dir=log_dir)  #训练日志
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'],optimizer=optimizer)
    model.fit(train_generator,validation_data=valid_generator,
                       epochs=epochs,callbacks=[tensorboard, early_stopping,checkpoint_period]
                       )
    predictions = model.predict(test_gen)
    np.save('/kaggle/working/predictions.npy', predictions)
    model.save('/kaggle/working/model.h5')
if __name__ == '__main__':
    train()
