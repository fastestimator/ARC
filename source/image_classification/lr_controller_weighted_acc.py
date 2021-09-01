from collections import deque

import cv2
import fastestimator as fe
import numpy as np
import tensorflow as tf
from fastestimator.backend import feed_forward, get_lr, set_lr
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, PadIfNeeded, RandomCrop
from fastestimator.op.numpyop.univariate import CoarseDropout, Normalize
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.metric import Accuracy
from tensorflow.keras import layers


def lstm_stacked(input_shape=(300, 3), num_classes=3):
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.LSTM(64, return_sequences=True))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


def zscore(data, epsilon=1e-7):
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean) / max(std, epsilon)
    return data


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


def preprocess_train_loss(train_loss, missing):
    target_size = (3 - missing) * 100
    train_loss = np.array(train_loss, dtype="float32")
    train_loss = cv2.resize(train_loss, (1, target_size))
    train_loss = zscore(train_loss)
    if train_loss.size < 300:
        train_loss = np.pad(train_loss, ((300 - train_loss.size, 0), (0, 0)), mode='constant', constant_values=0.0)
    return train_loss


def preprocess_val_loss(val_loss):
    val_loss = zscore(val_loss)
    val_loss = cv2.resize(val_loss, (1, 300), interpolation=cv2.INTER_NEAREST)
    return val_loss


def preprocess_train_lr(train_lr):
    train_lr = train_lr / train_lr[-1]
    train_lr = cv2.resize(train_lr, (1, 300), interpolation=cv2.INTER_NEAREST)
    return train_lr


class LRController(fe.trace.Trace):
    def __init__(self, model, controller, control_frequency=1):
        super().__init__(mode="train")
        self.model = model
        self.controller = controller
        self.control_frequency = control_frequency
        self.lr_multiplier = {0: 1.618, 1: 1.0, 2: 0.618}
        self.cycle_loss = []
        self.all_train_loss = deque([None] * 3, maxlen=3)
        self.all_train_lr = deque([None] * 3, maxlen=3)

    def on_batch_end(self, data):
        self.cycle_loss.append(data["ce"].numpy())
        # add lr in the log to help debugging
        if self.system.global_step % self.system.log_steps == 0 or self.system.global_step == 1:
            data.write_with_log("model_lr", np.float32(get_lr(self.model)))

    def on_epoch_begin(self, data):
        if (self.system.epoch_idx % self.control_frequency == 1
                or self.control_frequency == 1) and self.system.epoch_idx > 1:
            # preprocessing
            train_loss, missing = merge_list(self.all_train_loss)
            train_loss = preprocess_train_loss(train_loss, missing)
            val_loss, _ = merge_list(self.system.all_val_loss)
            val_loss = preprocess_val_loss(val_loss)
            train_lr, _ = merge_list(self.all_train_lr)
            train_lr = preprocess_train_lr(train_lr)
            model_inputs = np.concatenate((train_loss, val_loss, train_lr), axis=1)
            model_inputs = np.expand_dims(model_inputs, axis=0)
            # prediction
            model_pred = feed_forward(model=self.controller, x=model_inputs, training=False)
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
    def __init__(self, control_frequency=1, use_val_loss=True):
        super().__init__(mode="eval")
        self.control_frequency = control_frequency
        self.cycle_val_loss = []
        self.all_val_loss = deque([None] * 3, maxlen=3)
        self.use_val_loss = use_val_loss

    def on_epoch_end(self, data):
        if self.use_val_loss:
            self.cycle_val_loss.append(data["ce"])
        else:
            self.cycle_val_loss.append(0.0)
        if self.system.epoch_idx % self.control_frequency == 0:
            self.all_val_loss.append(self.cycle_val_loss)
            self.cycle_val_loss = []
            # use system instance to communicate between train and eval traces (work around)
            self.system.all_val_loss = self.all_val_loss


def residual(x, num_channel):
    x = layers.Conv2D(num_channel, 3, padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(num_channel, 3, padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    return x


def my_model():
    # prep layers
    inp = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(64, 3, padding='same')(inp)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    # layer1
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Add()([x, residual(x, 128)])
    # layer2
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    # layer3
    x = layers.Conv2D(512, 3, padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Add()([x, residual(x, 512)])
    # layers4
    x = layers.GlobalMaxPool2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(10)(x)
    x = layers.Activation('softmax', dtype='float32')(x)
    model = tf.keras.Model(inputs=inp, outputs=x)
    return model


def get_estimator(init_lr,
                  epochs=30,
                  batch_size=128,
                  control_frequency=3,
                  weights_path="../../model/model_best_wacc.h5"):
    # step 1
    train_data, eval_data = fe.dataset.data.cifar10.load_data()
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            PadIfNeeded(min_height=40, min_width=40, image_in="x", image_out="x", mode="train"),
            RandomCrop(32, 32, image_in="x", image_out="x", mode="train"),
            Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train")),
            CoarseDropout(inputs="x", outputs="x", mode="train", max_holes=1)
        ])
    # step 2
    model = fe.build(model_fn=my_model, optimizer_fn=lambda: tf.optimizers.Adam(init_lr))
    controller = fe.build(model_fn=lstm_stacked, optimizer_fn=None, weights_path=weights_path)
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    # step 3
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        LRController(model=model, controller=controller, control_frequency=control_frequency),
        RecordValLoss(control_frequency=control_frequency, use_val_loss=True)
    ]
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
    return estimator
