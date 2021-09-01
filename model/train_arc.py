import os
import pickle

import cv2
import fastestimator as fe
import numpy as np
import tensorflow as tf
from fastestimator.op.numpyop import Delete
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import ModelSaver
from tensorflow.python.keras import layers


def zscore(data, epsilon=1e-7):
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean) / max(std, epsilon)
    return data

def load_pickle(pickle_path):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data

def pad_left_one(data):
    data_length = data.size
    if data_length < 300:
        data = np.pad(data, ((300 - data_length, 0), (0, 0)),
                      mode='constant',
                      constant_values=1.0)
    return data

def pad_left_zero(data):
    data_length = data.size
    if data_length < 300:
        data = np.pad(data, ((300 - data_length, 0), (0, 0)),
                      mode='constant',
                      constant_values=0.0)
    return data

class RemoveValLoss(fe.op.numpyop.NumpyOp):
    def forward(self, data, state):
        val_loss = data
        return np.zeros_like(val_loss)

class MultipleClsBinaryCELoss(fe.op.tensorop.TensorOp):
    def __init__(self,
                 inputs,
                 outputs,
                 pos_labels=[0, 2],
                 neg_labels=[1],
                 mode=None):
        self.pos_labels = pos_labels
        self.neg_labels = neg_labels
        self.all_labels = self.pos_labels + self.neg_labels
        self.missing_labels = list(set([0, 1, 2]) - set(self.all_labels))
        if len(self.missing_labels) == 0:
            self.missing_labels = [-1]

        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    def forward(self, data, state):
        cls_pred, cls_label = data

        batch_size = cls_label.shape[0]
        binaryCEloss = 0.0

        case_count = 0.0
        for idx in range(batch_size):
            if cls_label[idx] != self.missing_labels[0]:
                abnormal_predict = tf.clip_by_value(
                    tf.math.reduce_max(
                        [cls_pred[idx][p] for p in self.pos_labels]), 1e-4,
                    1.0 - 1e-4)

                if cls_label[idx] != self.neg_labels[0]:
                    abnormal_label = 1.0
                else:
                    abnormal_label = 0.0

                binaryCEloss -= (
                    abnormal_label * tf.math.log(abnormal_predict) +
                    (1.0 - abnormal_label) *
                    tf.math.log(1.0 - abnormal_predict))
                case_count += 1

        return binaryCEloss / case_count

class CombineLoss(fe.op.tensorop.TensorOp):
    def __init__(self, inputs, outputs, weights, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.weights = weights

    def forward(self, data, state):
        return tf.reduce_sum(
            [loss * weight for loss, weight in zip(data, self.weights)])

class CombineData(fe.op.numpyop.NumpyOp):
    def forward(self, data, state):
        x = np.concatenate(data, axis=1)
        return np.float32(x)

class PreprocessTrainLoss(fe.op.numpyop.NumpyOp):
    def forward(self, data, state):
        train_loss = data
        train_loss = zscore(train_loss)
        train_loss = pad_left_zero(train_loss)
        return train_loss

class PreprocessTrainLR(fe.op.numpyop.NumpyOp):
    def forward(self, data, state):
        train_lr = data
        train_lr = train_lr / train_lr[0]
        train_lr = cv2.resize(train_lr, (1, train_lr.size * 100),
                              interpolation=cv2.INTER_NEAREST)
        train_lr = pad_left_one(train_lr)
        return train_lr

class PreprocessValLoss(fe.op.numpyop.NumpyOp):
    def forward(self, data, state):
        val_loss, train_loss = data
        val_loss = zscore(val_loss)
        val_loss = cv2.resize(val_loss, (1, train_loss.size),
                              interpolation=cv2.INTER_NEAREST)
        val_loss = pad_left_zero(val_loss)
        return val_loss

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


def get_estimator(epochs=300, data_dir="../data/offline_data.pkl", save_dir="./"):
    ckpt_save_dir = os.path.join(save_dir, "checkpoint")
    os.makedirs(ckpt_save_dir, exist_ok=True)
    train_ds = fe.dataset.NumpyDataset(data=load_pickle(data_dir))
    pipeline = fe.Pipeline(train_data=train_ds,
                           batch_size=128,
                           ops=[
                               PreprocessValLoss(inputs=["val_loss", "train_loss"],
                                                 outputs="val_loss"),
                               PreprocessTrainLoss(inputs="train_loss",
                                                   outputs="train_loss"),
                               PreprocessTrainLR(inputs="train_lr",
                                                 outputs="train_lr"),
                               Sometimes(RemoveValLoss(inputs="val_loss",
                                                       outputs="val_loss",
                                                       mode="train"),
                                         prob=0.5),
                               CombineData(inputs=("train_loss", "val_loss",
                                                   "train_lr"),
                                           outputs="x"),
                               Delete(keys=("train_loss", "val_loss",
                                            "train_lr"))
                           ])
    model = fe.build(model_fn=lstm_stacked,
                     optimizer_fn=lambda: tf.optimizers.Adam(1e-4))
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "label"), outputs="ce"),
        MultipleClsBinaryCELoss(inputs=("y_pred", "label"),
                                pos_labels=[0],
                                neg_labels=[2],
                                outputs="bnce_0vs2"),
        CombineLoss(inputs=["ce", "bnce_0vs2"],
                    outputs="total_loss",
                    weights=[0.5, 0.5]),
        UpdateOp(model=model, loss_name="total_loss")
    ])
    traces = [
        ModelSaver(model=model, save_dir=ckpt_save_dir, frequency=1, max_to_keep=1)
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             log_steps=5)
    return estimator
