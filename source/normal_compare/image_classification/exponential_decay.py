import fastestimator as fe
import tensorflow as tf
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, PadIfNeeded, RandomCrop
from fastestimator.op.numpyop.univariate import CoarseDropout, Normalize
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.metric import Accuracy
from tensorflow.python.keras import layers


def exponential_decay(time, init_lr, gamma=0.9, start=1):
    return init_lr * gamma**(time - start)


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


def get_estimator(init_lr, epochs=30, batch_size=128):
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
    model = fe.build(model_fn=my_model, optimizer_fn=lambda: tf.optimizers.Adam(init_lr))  # 1e-2, 1e-3, 1e-4
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    # step 3
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        LRScheduler(model=model, lr_fn=lambda epoch: exponential_decay(epoch, init_lr=init_lr))
    ]
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
    return estimator
