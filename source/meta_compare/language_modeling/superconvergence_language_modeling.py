import os

import fastestimator as fe
import numpy as np
import tensorflow as tf
import wget
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import cosine_decay
from fastestimator.trace.adapt import LRScheduler


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


def super_schedule(step, lr_max=23.2, lr_min=1.0, steps_per_epoch=345):
    if step < steps_per_epoch * 45:
        lr = (lr_max - lr_min) / steps_per_epoch / 45 * step + lr_min
    elif step < steps_per_epoch * 90:
        lr = lr_max - (step - steps_per_epoch * 45) / (steps_per_epoch * 90 - steps_per_epoch * 45) * (lr_max - lr_min)
    else:
        lr = lr_min - (step - steps_per_epoch * 90) / (steps_per_epoch * 98 - steps_per_epoch * 90) * (lr_min - 0.0)
    return lr


def get_estimator(data_dir, seq_length=20, batch_size=128, vocab_size=10000, epochs=98):
    train_data, _, test_data = get_ptb(folder_path=data_dir, seq_length=seq_length + 1)
    pipeline = fe.Pipeline(train_data=fe.dataset.NumpyDataset(data={"x": train_data}),
                           eval_data=fe.dataset.NumpyDataset(data={"x": test_data}),
                           batch_size=batch_size,
                           ops=CreateInputAndTarget(inputs="x", outputs=("x", "y")),
                           drop_last=True)
    # step 2
    model = fe.build(model_fn=lambda: build_model(vocab_size, embedding_dim=300, rnn_units=600, batch_size=batch_size),
                     optimizer_fn=lambda: tf.optimizers.SGD(1.8, momentum=0.9))
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        SparseCrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    # step 3
    traces = [
        Perplexity(inputs="ce", outputs="perplexity", mode="eval"),
        LRScheduler(model=model, lr_fn=lambda step: super_schedule(step))
    ]

    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
    return estimator
