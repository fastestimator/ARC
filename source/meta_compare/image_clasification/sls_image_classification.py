import fastestimator as fe
import sls
import torch
import torch.nn as nn
import torch.nn.functional as fn
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, PadIfNeeded, RandomCrop
from fastestimator.op.numpyop.univariate import ChannelTranspose, CoarseDropout, Normalize
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.metric import Accuracy


class DummpyUpdate(UpdateOp):
    def forward(self, data, state):
        pass


class FastCifar(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 64, 3, padding=(1, 1))
        self.conv0_bn = nn.BatchNorm2d(64, momentum=0.8)
        self.conv1 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        self.conv1_bn = nn.BatchNorm2d(128, momentum=0.8)
        self.residual1 = Residual(128, 128)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=(1, 1))
        self.conv2_bn = nn.BatchNorm2d(256, momentum=0.8)
        self.residual2 = Residual(256, 256)
        self.conv3 = nn.Conv2d(256, 512, 3, padding=(1, 1))
        self.conv3_bn = nn.BatchNorm2d(512, momentum=0.8)
        self.residual3 = Residual(512, 512)
        self.fc1 = nn.Linear(512, 10)

    def forward(self, x):
        # prep layer
        x = self.conv0(x)
        x = self.conv0_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        # layer 1
        x = self.conv1(x)
        x = fn.max_pool2d(x, 2)
        x = self.conv1_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        x = x + self.residual1(x)
        # layer 2
        x = self.conv2(x)
        x = fn.max_pool2d(x, 2)
        x = self.conv2_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        x = x + self.residual2(x)
        # layer 3
        x = self.conv3(x)
        x = fn.max_pool2d(x, 2)
        x = self.conv3_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        x = x + self.residual3(x)
        # layer 4
        # Storing kernel size as a list in case the user needs to export the model to ONNX
        # As ONNX doesn't support dynamic kernel size
        size_array = [int(s) for s in x.size()[2:]]
        x = fn.max_pool2d(x, kernel_size=size_array)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = fn.softmax(x, dim=-1)
        return x


class Residual(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out, 3, padding=(1, 1))
        self.conv1_bn = nn.BatchNorm2d(channel_out)
        self.conv2 = nn.Conv2d(channel_out, channel_out, 3, padding=(1, 1))
        self.conv2_bn = nn.BatchNorm2d(channel_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        return x


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


def get_estimator(epochs=30, batch_size=128):
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
            CoarseDropout(inputs="x", outputs="x", mode="train", max_holes=1),
            ChannelTranspose(inputs="x", outputs="x")
        ])
    # step 2
    model = fe.build(model_fn=FastCifar, optimizer_fn="sgd")
    opt = sls.Sls(model.parameters())
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        SGDLinesSearch(model=model,
                       opt=opt,
                       loss_op=CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
                       inputs=("x", "y"),
                       outputs="ce"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce", mode="eval"),
        DummpyUpdate(model=model, loss_name="ce")
    ])
    # step 3
    traces = [Accuracy(true_key="y", pred_key="y_pred"), PrintLR(opt=opt)]
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
    return estimator
