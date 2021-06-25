"""
This is the proxy test training code for ARC model selection.

The overall proxy test steps include:
1. Use this code to trained a WideResNet on SVHN_Cropped dataset and record the best eval acc.
2. Repeat step 1 with 3 different learning rate [0.1, 0.001, 0.00001] for 5 times each.
3. The average over 15 eval acc will be the ARC model's score.

The best ARC model is the one with highest score.
"""

import os
from collections import deque

import cv2
import fastestimator as fe
import numpy as np
import tensorflow as tf
from fastestimator.backend import feed_forward, get_lr, set_lr
from fastestimator.dataset.data import svhn_cropped
from fastestimator.op.numpyop.univariate import CoarseDropout, ReadImage
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.metric import Accuracy
from tensorflow.keras import Model, layers


def merge_list(data):
    output = []
    missing = 0
    for item in data:
        if isinstance(item, list):
            output.extend(item)
        elif item:
            output.append(item)
        else:
            missing += 1
    return output, missing


def zscore(data, epsilon=1e-7):
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean) / max(std, epsilon)
    return data


def preprocess_train_loss(train_loss, missing):
    target_size = (3 - missing) * 100
    train_loss = np.array(train_loss, dtype="float32")
    train_loss = cv2.resize(train_loss, (1, target_size))
    train_loss = zscore(train_loss)
    if train_loss.size < 300:
        train_loss = np.pad(train_loss, ((300 - train_loss.size, 0), (0, 0)),
                            mode='constant',
                            constant_values=0.0)
    return train_loss


def preprocess_val_loss(val_loss, missing):
    target_size = (3 - missing) * 100
    val_loss = zscore(val_loss)
    val_loss = cv2.resize(val_loss, (1, target_size),
                          interpolation=cv2.INTER_NEAREST)

    if val_loss.size < 300:
        val_loss = np.pad(val_loss, ((300 - val_loss.size, 0), (0, 0)),
                          mode='constant',
                          constant_values=0.0)
    return val_loss


def preprocess_train_lr(train_lr, missing):
    target_size = (3 - missing) * 100
    train_lr = train_lr / train_lr[0]  # different

    train_lr = cv2.resize(train_lr, (1, target_size),
                          interpolation=cv2.INTER_NEAREST)
    if train_lr.size < 300:
        train_lr = np.pad(train_lr, ((300 - train_lr.size, 0), (0, 0)),
                          mode='constant',
                          constant_values=1.0)  # different
    return train_lr


class LRController(fe.trace.Trace):
    def __init__(self, model, controller, control_frequency=1, loss_key="ce"):
        super().__init__(mode="train")
        self.model = model
        self.controller = controller
        self.control_frequency = control_frequency
        self.lr_multiplier = {0: 1.618, 1: 1.0, 2: 0.618}
        self.cycle_loss = []
        self.all_train_loss = deque([None] * 3, maxlen=3)
        self.all_train_lr = deque([None] * 3, maxlen=3)
        self.loss_key = loss_key

    def on_batch_end(self, data):
        self.cycle_loss.append(data[self.loss_key].numpy())
        # add lr in the log to help debugging
        if self.system.global_step % self.system.log_steps == 0 or self.system.global_step == 1:
            data.write_with_log("model_lr", np.float32(get_lr(self.model)))

    def on_epoch_begin(self, data):
        if (self.system.epoch_idx % self.control_frequency == 1
                or self.control_frequency == 1) and self.system.epoch_idx > 1:
            # preprocessing
            train_loss, missing = merge_list(self.all_train_loss)
            train_loss = preprocess_train_loss(train_loss, missing)
            val_loss, missing = merge_list(self.system.all_val_loss)
            val_loss = preprocess_val_loss(val_loss, missing)
            train_lr, missing = merge_list(self.all_train_lr)
            train_lr = preprocess_train_lr(train_lr, missing)
            model_inputs = np.concatenate((train_loss, val_loss, train_lr),
                                          axis=1)
            model_inputs = np.expand_dims(model_inputs, axis=0)
            # prediction
            model_pred = feed_forward(model=self.controller,
                                      x=model_inputs,
                                      training=False)
            action = np.argmax(model_pred)
            multiplier = self.lr_multiplier[action]
            current_lr = get_lr(model=self.model)
            set_lr(model=self.model, lr=current_lr * multiplier)
            print("multiplying lr by {}".format(multiplier))

    def on_epoch_end(self, data):
        if self.system.epoch_idx % self.control_frequency == 0:
            current_lr = get_lr(model=self.model)
            self.all_train_loss.append(self.cycle_loss)
            self.all_train_lr.append(current_lr)
            self.cycle_loss = []


class RecordValLoss(fe.trace.Trace):
    def __init__(self, control_frequency=1, use_val_loss=True, loss_key="ce"):
        super().__init__(mode="eval")
        self.control_frequency = control_frequency
        self.cycle_val_loss = []
        self.all_val_loss = deque([None] * 3, maxlen=3)
        self.use_val_loss = use_val_loss
        self.loss_key = loss_key

    def on_epoch_end(self, data):
        if self.use_val_loss:
            self.cycle_val_loss.append(data[self.loss_key])
        else:
            self.cycle_val_loss.append(0.0)
        if self.system.epoch_idx % self.control_frequency == 0:
            self.all_val_loss.append(self.cycle_val_loss)
            self.cycle_val_loss = []
            # use system instance to communicate between train and eval traces (work around)
            self.system.all_val_loss = self.all_val_loss


def lstm_stacked(input_shape=(300, 3), num_classes=3):
    model = tf.keras.Sequential()
    model.add(
        layers.Conv1D(filters=32,
                      kernel_size=5,
                      activation='relu',
                      input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.LSTM(64, return_sequences=True))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


def WideResidualNetwork(input_shape,
                        depth=28,
                        width=8,
                        dropout_rate=0.0,
                        classes=10,
                        activation='softmax'):

    if (depth - 4) % 6 != 0:
        raise ValueError('Depth of the network must be such that (depth - 4)'
                         'should be divisible by 6.')

    img_input = layers.Input(shape=input_shape)

    x = __create_wide_residual_network(classes, img_input, True, depth, width,
                                       dropout_rate, activation)
    inputs = img_input
    # Create model.
    model = Model(inputs, x)
    return model


def __conv1_block(inputs):
    x = layers.Conv2D(16, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def __conv2_block(inputs, k=1, dropout=0.0):
    init = inputs

    # Check if input number of filters is same as 16 * k, else create
    # convolution2d for this input
    if init.shape[-1] != 16 * k:
        init = layers.Conv2D(16 * k, (1, 1),
                             activation='linear',
                             padding='same')(init)

    x = layers.Conv2D(16 * k, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    if dropout > 0.0:
        x = layers.Dropout(dropout)(x)

    x = layers.Conv2D(16 * k, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    m = init + x
    return m


def __conv3_block(inputs, k=1, dropout=0.0):
    init = inputs
    # Check if input number of filters is same as 32 * k, else
    # create convolution2d for this input
    if init.shape[-1] != 32 * k:
        init = layers.Conv2D(32 * k, (1, 1),
                             activation='linear',
                             padding='same')(init)
    x = layers.Conv2D(32 * k, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if dropout > 0.0:
        x = layers.Dropout(dropout)(x)
    x = layers.Conv2D(32 * k, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    m = init + x
    return m


def ___conv4_block(inputs, k=1, dropout=0.0):
    init = inputs
    if init.shape[-1] != 64 * k:
        init = layers.Conv2D(64 * k, (1, 1),
                             activation='linear',
                             padding='same')(init)
    x = layers.Conv2D(64 * k, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if dropout > 0.0:
        x = layers.Dropout(dropout)(x)
    x = layers.Conv2D(64 * k, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    m = init + x
    return m


def __create_wide_residual_network(nb_classes,
                                   img_input,
                                   include_top,
                                   depth=28,
                                   width=8,
                                   dropout=0.0,
                                   activation='softmax'):
    ''' Creates a Wide Residual Network with specified parameters
    Args:
        nb_classes: Number of output classes
        img_input: Input tensor or layer
        include_top: Flag to include the last dense layer
        depth: Depth of the network. Compute N = (n - 4) / 6.
               For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
               For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
               For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
        width: Width of the network.
        dropout: Adds dropout if value is greater than 0.0
    Returns:a Keras Model
    '''
    N = (depth - 4) // 6
    x = __conv1_block(img_input)
    nb_conv = 4
    for i in range(N):
        x = __conv2_block(x, width, dropout)
        nb_conv += 2

    x = layers.MaxPooling2D((2, 2))(x)

    for i in range(N):
        x = __conv3_block(x, width, dropout)
        nb_conv += 2

    x = layers.MaxPooling2D((2, 2))(x)

    for i in range(N):
        x = ___conv4_block(x, width, dropout)
        nb_conv += 2

    if include_top:
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(nb_classes, activation=activation)(x)
    return x


class Scale(fe.op.numpyop.NumpyOp):
    def forward(self, data, state):
        data = data / 255
        return data


def get_estimator(data_dir,
                  init_lr,
                  weights_path,
                  control_frequency=3,
                  batch_size=128,
                  epochs=30):
    train_ds, test_ds = svhn_cropped.load()
    pipeline = fe.Pipeline(train_data=train_ds,
                           eval_data=test_ds,
                           batch_size=batch_size,
                           ops=[
                               ReadImage(inputs="x", outputs="x"),
                               Scale(inputs="x", outputs="x"),
                               CoarseDropout(inputs="x",
                                             outputs="x",
                                             mode="train",
                                             max_holes=1)
                           ])
    model = fe.build(model_fn=lambda: WideResidualNetwork(
        input_shape=(32, 32, 3), depth=28, width=2),
                     optimizer_fn=lambda: tf.optimizers.Adam(init_lr))
    controller = fe.build(model_fn=lstm_stacked,
                          optimizer_fn=None,
                          weights_path=weights_path)
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        LRController(model=model,
                     controller=controller,
                     control_frequency=control_frequency),
        RecordValLoss(control_frequency=control_frequency, use_val_loss=True)
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces)
    return estimator
