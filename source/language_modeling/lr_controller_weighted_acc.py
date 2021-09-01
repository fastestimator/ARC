import os
from collections import deque

import cv2
import fastestimator as fe
import numpy as np
import tensorflow as tf
import wget
from fastestimator.backend import feed_forward, get_lr, set_lr
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from tensorflow.keras import layers


def get_ptb(folder_path, seq_length=64):
    file_names = ["ptb.train.txt", "ptb.valid.txt", "ptb.test.txt"]
    urls = [
        'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
        'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt',
        'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt'
    ]
    # Read text
    texts = []
    for file_name, url in zip(file_names, urls):
        text = []
        file_path = os.path.join(folder_path, file_name)
        if not os.path.exists(file_path):
            wget.download(url, out=folder_path)
        with open(file_path, 'r') as f:
            for line in f:
                text.extend(line.split() + ['<eos>'])
        texts.append(text)
    # Build dictionary from training data
    vocab = sorted(set(texts[0]))
    word2idx = {u: i for i, u in enumerate(vocab)}
    idx2word = np.array(vocab)

    #convert word to index and split the sequences and discard the last incomplete sequence
    data = [[word2idx[word] for word in text[:-(len(text) % seq_length)]] for text in texts]
    train_data, eval_data, test_data = [np.array(d).reshape(-1, seq_length) for d in data]
    return train_data, eval_data, test_data


class CreateInputAndTarget(fe.op.numpyop.NumpyOp):
    def forward(self, data, state):
        x = data
        return x[:-1], x[1:]


class SparseCrossEntropy(fe.op.tensorop.TensorOp):
    def forward(self, data, state):
        y_pred, y = data
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, y_pred, from_logits=True)
        return tf.reduce_mean(loss)


class Perplexity(fe.trace.Trace):
    def on_epoch_end(self, data):
        ce = data["ce"]
        data.write_with_log(self.outputs[0], np.exp(ce))


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


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


def get_estimator(init_lr,
                  data_dir,
                  seq_length=20,
                  batch_size=128,
                  vocab_size=10000,
                  epochs=98,
                  control_frequency=9,
                  weights_path="../../model/model_best_wacc.h5"):
    train_data, _, test_data = get_ptb(folder_path=data_dir, seq_length=seq_length + 1)
    pipeline = fe.Pipeline(train_data=fe.dataset.NumpyDataset(data={"x": train_data}),
                           eval_data=fe.dataset.NumpyDataset(data={"x": test_data}),
                           batch_size=batch_size,
                           ops=CreateInputAndTarget(inputs="x", outputs=("x", "y")),
                           drop_last=True)
    # step 2
    model = fe.build(model_fn=lambda: build_model(vocab_size, embedding_dim=300, rnn_units=600, batch_size=batch_size),
                     optimizer_fn=lambda: tf.optimizers.SGD(init_lr, momentum=0.9))  #1.0, 0.1, 0.01
    controller = fe.build(model_fn=lstm_stacked, optimizer_fn=None, weights_path=weights_path)
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        SparseCrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    # step 3
    traces = [
        Perplexity(inputs="ce", outputs="perplexity", mode="eval"),
        LRController(model=model, controller=controller, control_frequency=control_frequency),
        RecordValLoss(control_frequency=control_frequency, use_val_loss=True)
    ]
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
    return estimator
