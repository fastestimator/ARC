import os

import fastestimator as fe
import numpy as np
import sls
import torch
import torch.nn as nn
import wget
from fastestimator.op.numpyop import NumpyOp
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp


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
    #convert word to index and split the sequences and discard the last incomplete sequence
    data = [[word2idx[word] for word in text[:-(len(text) % seq_length)]] for text in texts]
    train_data, eval_data, test_data = [np.array(d).reshape(-1, seq_length) for d in data]
    return train_data, eval_data, test_data


class CreateInputAndTarget(NumpyOp):
    def forward(self, data, state):
        return data[:-1], data[1:]


class DimesionAdjust(TensorOp):
    def forward(self, data, state):
        x, y = data
        return x.T, y.T.reshape(-1)


class Perplexity(fe.trace.Trace):
    def on_epoch_end(self, data):
        ce = data["ce"]
        data.write_with_log(self.outputs[0], np.exp(ce))


class BuildModel(nn.Module):
    def __init__(self, vocab_size=10000, embedding_dim=300, rnn_units=600):
        super().__init__()
        self.embed_layer = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_layer = nn.LSTM(embedding_dim, rnn_units)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(rnn_units, vocab_size)

        nn.init.xavier_uniform_(self.lstm_layer.weight_ih_l0.data)
        nn.init.xavier_uniform_(self.lstm_layer.weight_hh_l0.data)

    def forward(self, x):
        x = self.embed_layer(x)
        x, _ = self.lstm_layer(x)
        x = x.view(x.size(0) * x.size(1), x.size(2))
        x = self.dropout(x)
        x = self.fc(x)
        return x


class DummpyUpdate(UpdateOp):
    def forward(self, data, state):
        pass


class SGDLinesSearch(fe.op.tensorop.TensorOp):
    def __init__(self, model, opt, loss_op, inputs, outputs, mode="train"):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.model = model
        self.opt = opt
        self.loss_op = loss_op

    def forward(self, data, state):
        x, y = data
        closure = lambda: self.loss_op.forward((self.model(x), y), state=state)
        self.opt.zero_grad()
        loss = self.opt.step(closure=closure)
        return loss


class PrintLR(fe.trace.Trace):
    def __init__(self, opt):
        super().__init__(mode="train")
        self.opt = opt

    def on_batch_end(self, data):
        if self.system.global_step % self.system.log_steps == 0 or self.system.global_step == 1:
            data.write_with_log("model_lr", float(self.opt.state['step_size']))


def get_estimator(data_dir, epochs=98, batch_size=128, seq_length=20, vocab_size=10000):
    train_data, _, test_data = get_ptb(folder_path=data_dir, seq_length=seq_length + 1)
    pipeline = fe.Pipeline(train_data=fe.dataset.NumpyDataset(data={"x": train_data}),
                           eval_data=fe.dataset.NumpyDataset(data={"x": test_data}),
                           batch_size=batch_size,
                           ops=CreateInputAndTarget(inputs="x", outputs=("x", "y")),
                           drop_last=True)
    # step 2
    model = fe.build(model_fn=lambda: BuildModel(vocab_size, embedding_dim=300, rnn_units=600), optimizer_fn="sgd")
    opt = sls.Sls(model.parameters())
    network = fe.Network(ops=[
        DimesionAdjust(inputs=("x", "y"), outputs=("x", "y")),
        ModelOp(model=model, inputs="x", outputs="y_pred", mode=None),
        SGDLinesSearch(model=model,
                       opt=opt,
                       loss_op=CrossEntropy(inputs=("y_pred", "y"), outputs="ce", form="sparse", from_logits=True),
                       inputs=("x", "y"),
                       outputs="ce"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce", form="sparse", from_logits=True, mode="eval"),
        DummpyUpdate(model=model, loss_name="ce")
    ])
    # step 3
    traces = [Perplexity(inputs="ce", outputs="perplexity", mode="eval"), PrintLR(opt=opt)]
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
    return estimator
