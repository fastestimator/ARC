import pickle
from typing import Set, Union

import cv2
import fastestimator as fe
import numpy as np
import tensorflow as tf
from fastestimator.op.numpyop import Delete
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.metric import Accuracy, ConfusionMatrix
from fastestimator.util import Data
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras import layers


class WeightedAccuracy(fe.trace.Trace):
    def __init__(self,
                 true_key: str,
                 pred_key: str,
                 mode: Union[str, Set[str]] = ("eval", "test"),
                 output_name=["wacc"]) -> None:
        super().__init__(inputs=(true_key, pred_key), outputs=output_name, mode=mode)
        self.num_classes = 3
        self.matrix = None
        self.rpm_abs = np.array([[3, 1, 3], [1, 1, 1], [3, 1, 3]])

    @property
    def true_key(self) -> str:
        return self.inputs[0]

    @property
    def pred_key(self) -> str:
        return self.inputs[1]

    def on_epoch_begin(self, data: Data) -> None:
        self.matrix = None

    def on_batch_end(self, data: Data) -> None:
        y_true, y_pred = data[self.true_key].numpy(), data[self.pred_key].numpy()
        if y_true.shape[-1] > 1 and y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=-1)
        if y_pred.shape[-1] > 1:
            y_pred = np.argmax(y_pred, axis=-1)
        else:
            y_pred = np.round(y_pred)
        assert y_pred.size == y_true.size
        batch_confusion = confusion_matrix(y_true, y_pred, labels=list(range(0, self.num_classes)))
        if self.matrix is None:
            self.matrix = batch_confusion
        else:
            self.matrix += batch_confusion

    def on_epoch_end(self, data: Data) -> None:
        numerator = self.rpm_abs[0, 0] * self.matrix[0, 0] + self.rpm_abs[1, 1] * self.matrix[1, 1] + self.rpm_abs[
            2, 2] * self.matrix[2, 2]
        denominator = np.sum(self.matrix * self.rpm_abs)
        data.write_with_log(self.outputs[0], numerator / denominator)


def zscore(data, epsilon=1e-7):
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean) / max(std, epsilon)
    return data


def load_pickle(pickle_path):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data


class CombineData(fe.op.numpyop.NumpyOp):
    def forward(self, data, state):
        train_loss, val_loss, train_lr = data
        x = np.concatenate((train_loss, val_loss, train_lr), axis=1)
        return np.float32(x)


class PreprocessTrainLoss(fe.op.numpyop.NumpyOp):
    def forward(self, data, state):
        train_loss = data
        train_loss = zscore(train_loss)
        train_loss = self.pad_left_zero(train_loss)
        return train_loss

    def pad_left_zero(self, data):
        data_length = data.size
        if data_length < 300:
            data = np.pad(data, ((300 - data_length, 0), (0, 0)), mode='constant', constant_values=0.0)
        return data


class PreprocessValLoss(fe.op.numpyop.NumpyOp):
    def forward(self, data, state):
        val_loss = data
        val_loss = zscore(val_loss)
        val_loss = cv2.resize(val_loss, (1, 300), interpolation=cv2.INTER_NEAREST)
        return val_loss


class PreprocessTrainLR(fe.op.numpyop.NumpyOp):
    def forward(self, data, state):
        train_lr = data
        train_lr = train_lr / train_lr[-1]
        train_lr = cv2.resize(train_lr, (1, 300), interpolation=cv2.INTER_NEAREST)
        return train_lr


class RemoveValLoss(fe.op.numpyop.NumpyOp):
    def forward(self, data, state):
        val_loss = data
        return np.zeros_like(val_loss)


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


def get_estimator(data_dir="../data/offline_data.pkl"):
    train_ds = fe.dataset.NumpyDataset(data=load_pickle(data_dir))
    eval_ds = train_ds.split(0.3)
    pipeline = fe.Pipeline(
        train_data=train_ds,
        eval_data=eval_ds,
        batch_size=128,
        ops=[
            PreprocessTrainLoss(inputs="train_loss", outputs="train_loss"),
            PreprocessValLoss(inputs="val_loss", outputs="val_loss"),
            PreprocessTrainLR(inputs="train_lr", outputs="train_lr"),
            Sometimes(RemoveValLoss(inputs="val_loss", outputs="val_loss", mode="train"), prob=0.5),
            CombineData(inputs=("train_loss", "val_loss", "train_lr"), outputs="x"),
            Delete(keys=("train_loss", "val_loss", "train_lr"))
        ])
    model = fe.build(model_fn=lstm_stacked, optimizer_fn=lambda: tf.optimizers.Adam(1e-4))
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "label"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    traces = [
        Accuracy(true_key="label", pred_key="y_pred"),
        WeightedAccuracy(true_key="label", pred_key="y_pred"),
        ConfusionMatrix(true_key="label", pred_key="y_pred", num_classes=3)
    ]
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=300, traces=traces, log_steps=5)
    return estimator
