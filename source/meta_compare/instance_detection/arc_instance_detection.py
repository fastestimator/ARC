from collections import deque

import cv2
import fastestimator as fe
import numpy as np
import tensorflow as tf
from albumentations import BboxParams
from fastestimator.backend import feed_forward, get_lr, set_lr
from fastestimator.dataset.data import mscoco
from fastestimator.op.numpyop import NumpyOp
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, LongestMaxSize, PadIfNeeded
from fastestimator.op.numpyop.univariate import Normalize, ReadImage
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.metric import MeanAveragePrecision
from tensorflow.keras import layers, models, regularizers


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
    model.add(layers.Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.LSTM(64, return_sequences=True))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


def _get_fpn_anchor_box(width: int, height: int):
    assert height % 32 == 0 and width % 32 == 0
    shapes = [(int(height / 8), int(width / 8))]  # P3
    num_pixel = [np.prod(shapes)]
    anchor_lengths = [32, 64, 128, 256, 512]
    for _ in range(4):  # P4 through P7
        shapes.append((int(np.ceil(shapes[-1][0] / 2)), int(np.ceil(shapes[-1][1] / 2))))
        num_pixel.append(np.prod(shapes[-1]))
    total_num_pixels = np.sum(num_pixel)
    anchorbox = np.zeros((9 * total_num_pixels, 4))
    anchor_length_multipliers = [2**(0.0), 2**(1 / 3), 2**(2 / 3)]
    aspect_ratios = [1.0, 2.0, 0.5]  # x:y
    anchor_idx = 0
    for shape, anchor_length in zip(shapes, anchor_lengths):
        p_h, p_w = shape
        base_y = 2**np.ceil(np.log2(height / p_h))
        base_x = 2**np.ceil(np.log2(width / p_w))
        for i in range(p_h):
            center_y = (i + 1 / 2) * base_y
            for j in range(p_w):
                center_x = (j + 1 / 2) * base_x
                for anchor_length_multiplier in anchor_length_multipliers:
                    area = (anchor_length * anchor_length_multiplier)**2
                    for aspect_ratio in aspect_ratios:
                        x1 = center_x - np.sqrt(area * aspect_ratio) / 2
                        y1 = center_y - np.sqrt(area / aspect_ratio) / 2
                        x2 = center_x + np.sqrt(area * aspect_ratio) / 2
                        y2 = center_y + np.sqrt(area / aspect_ratio) / 2
                        anchorbox[anchor_idx, 0] = x1
                        anchorbox[anchor_idx, 1] = y1
                        anchorbox[anchor_idx, 2] = x2 - x1
                        anchorbox[anchor_idx, 3] = y2 - y1
                        anchor_idx += 1
        if p_h == 1 and p_w == 1:  # the next level of 1x1 feature map is still 1x1, therefore ignore
            break
    return np.float32(anchorbox), np.int32(num_pixel) * 9


class ShiftLabel(NumpyOp):
    def forward(self, data, state):
        # the label of COCO dataset starts from 1, shifting the start to 0
        bbox = np.array(data, dtype=np.float32)
        bbox[:, -1] = bbox[:, -1] - 1
        return bbox


class AnchorBox(NumpyOp):
    def __init__(self, width, height, inputs, outputs, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.anchorbox, _ = _get_fpn_anchor_box(width, height)  # anchorbox is #num_anchor x 4

    def forward(self, data, state):
        target = self._generate_target(data)  # bbox is #obj x 5
        return np.float32(target)

    def _generate_target(self, bbox):
        object_boxes = bbox[:, :-1]  # num_obj x 4
        label = bbox[:, -1]  # num_obj x 1
        ious = self._get_iou(object_boxes, self.anchorbox)  # num_obj x num_anchor
        # now for each object in image, assign the anchor box with highest iou to them
        anchorbox_best_iou_idx = np.argmax(ious, axis=1)
        num_obj = ious.shape[0]
        for row in range(num_obj):
            ious[row, anchorbox_best_iou_idx[row]] = 0.99
        # next, begin the anchor box assignment based on iou
        anchor_to_obj_idx = np.argmax(ious, axis=0)  # num_anchor x 1
        anchor_best_iou = np.max(ious, axis=0)  # num_anchor x 1
        cls_gt = np.int32([label[idx] for idx in anchor_to_obj_idx])  # num_anchor x 1
        cls_gt[np.where(anchor_best_iou <= 0.4)] = -1  # background class
        cls_gt[np.where(np.logical_and(anchor_best_iou > 0.4, anchor_best_iou <= 0.5))] = -2  # ignore these examples
        # finally, calculate localization target
        single_loc_gt = object_boxes[anchor_to_obj_idx]  # num_anchor x 4
        gt_x1, gt_y1, gt_width, gt_height = np.split(single_loc_gt, 4, axis=1)
        ac_x1, ac_y1, ac_width, ac_height = np.split(self.anchorbox, 4, axis=1)
        dx1 = np.squeeze((gt_x1 - ac_x1) / ac_width)
        dy1 = np.squeeze((gt_y1 - ac_y1) / ac_height)
        dwidth = np.squeeze(np.log(gt_width / ac_width))
        dheight = np.squeeze(np.log(gt_height / ac_height))
        return np.array([dx1, dy1, dwidth, dheight, cls_gt]).T  # num_anchor x 5

    @staticmethod
    def _get_iou(boxes1, boxes2):
        """Computes the value of intersection over union (IoU) of two array of boxes.
        Args:
            box1 (array): first boxes in N x 4
            box2 (array): second box in M x 4
        Returns:
            float: IoU value in N x M
        """
        x11, y11, w1, h1 = np.split(boxes1, 4, axis=1)
        x21, y21, w2, h2 = np.split(boxes2, 4, axis=1)
        x12 = x11 + w1
        y12 = y11 + h1
        x22 = x21 + w2
        y22 = y21 + h2
        xmin = np.maximum(x11, np.transpose(x21))
        ymin = np.maximum(y11, np.transpose(y21))
        xmax = np.minimum(x12, np.transpose(x22))
        ymax = np.minimum(y12, np.transpose(y22))
        inter_area = np.maximum((xmax - xmin + 1), 0) * np.maximum((ymax - ymin + 1), 0)
        area1 = (w1 + 1) * (h1 + 1)
        area2 = (w2 + 1) * (h2 + 1)
        iou = inter_area / (area1 + area2.T - inter_area)
        return iou


def _classification_sub_net(num_classes, num_anchor=9):
    model = models.Sequential()
    model.add(
        layers.Conv2D(256,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      activation='relu',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    model.add(
        layers.Conv2D(256,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      activation='relu',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    model.add(
        layers.Conv2D(256,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      activation='relu',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    model.add(
        layers.Conv2D(256,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      activation='relu',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    model.add(
        layers.Conv2D(num_classes * num_anchor,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      activation='sigmoid',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                      bias_initializer=tf.initializers.constant(np.log(1 / 99))))
    model.add(layers.Reshape((-1, num_classes)))  # the output dimension is [batch, #anchor, #classes]
    return model


def _regression_sub_net(num_anchor=9):
    model = models.Sequential()
    model.add(
        layers.Conv2D(256,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      activation='relu',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    model.add(
        layers.Conv2D(256,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      activation='relu',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    model.add(
        layers.Conv2D(256,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      activation='relu',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    model.add(
        layers.Conv2D(256,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      activation='relu',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    model.add(
        layers.Conv2D(4 * num_anchor,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    model.add(layers.Reshape((-1, 4)))  # the output dimension is [batch, #anchor, 4]
    return model


def RetinaNet(input_shape, num_classes, num_anchor=9):
    inputs = tf.keras.Input(shape=input_shape)
    # FPN
    resnet50 = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_tensor=inputs, pooling=None)
    assert resnet50.layers[80].name == "conv3_block4_out"
    C3 = resnet50.layers[80].output
    assert resnet50.layers[142].name == "conv4_block6_out"
    C4 = resnet50.layers[142].output
    assert resnet50.layers[-1].name == "conv5_block3_out"
    C5 = resnet50.layers[-1].output
    P5 = layers.Conv2D(256, kernel_size=1, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.0001))(C5)
    P5_upsampling = layers.UpSampling2D()(P5)
    P4 = layers.Conv2D(256, kernel_size=1, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.0001))(C4)
    P4 = layers.Add()([P5_upsampling, P4])
    P4_upsampling = layers.UpSampling2D()(P4)
    P3 = layers.Conv2D(256, kernel_size=1, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.0001))(C3)
    P3 = layers.Add()([P4_upsampling, P3])
    P6 = layers.Conv2D(256,
                       kernel_size=3,
                       strides=2,
                       padding='same',
                       name="P6",
                       kernel_regularizer=regularizers.l2(0.0001))(C5)
    P7 = layers.Activation('relu')(P6)
    P7 = layers.Conv2D(256,
                       kernel_size=3,
                       strides=2,
                       padding='same',
                       name="P7",
                       kernel_regularizer=regularizers.l2(0.0001))(P7)
    P5 = layers.Conv2D(256,
                       kernel_size=3,
                       strides=1,
                       padding='same',
                       name="P5",
                       kernel_regularizer=regularizers.l2(0.0001))(P5)
    P4 = layers.Conv2D(256,
                       kernel_size=3,
                       strides=1,
                       padding='same',
                       name="P4",
                       kernel_regularizer=regularizers.l2(0.0001))(P4)
    P3 = layers.Conv2D(256,
                       kernel_size=3,
                       strides=1,
                       padding='same',
                       name="P3",
                       kernel_regularizer=regularizers.l2(0.0001))(P3)
    # classification subnet
    cls_subnet = _classification_sub_net(num_classes=num_classes, num_anchor=num_anchor)
    P3_cls = cls_subnet(P3)
    P4_cls = cls_subnet(P4)
    P5_cls = cls_subnet(P5)
    P6_cls = cls_subnet(P6)
    P7_cls = cls_subnet(P7)
    cls_output = layers.Concatenate(axis=-2)([P3_cls, P4_cls, P5_cls, P6_cls, P7_cls])
    # localization subnet
    loc_subnet = _regression_sub_net(num_anchor=num_anchor)
    P3_loc = loc_subnet(P3)
    P4_loc = loc_subnet(P4)
    P5_loc = loc_subnet(P5)
    P6_loc = loc_subnet(P6)
    P7_loc = loc_subnet(P7)
    loc_output = layers.Concatenate(axis=-2)([P3_loc, P4_loc, P5_loc, P6_loc, P7_loc])
    return tf.keras.Model(inputs=inputs, outputs=[cls_output, loc_output])


class RetinaLoss(TensorOp):
    def forward(self, data, state):
        anchorbox, cls_pred, loc_pred = data
        batch_size = anchorbox.shape[0]
        focal_loss, l1_loss, total_loss = [], [], []
        for idx in range(batch_size):
            single_loc_gt, single_cls_gt = anchorbox[idx][:, :-1], tf.cast(anchorbox[idx][:, -1], tf.int32)
            single_loc_pred, single_cls_pred = loc_pred[idx], cls_pred[idx]
            single_focal_loss, anchor_obj_idx = self.focal_loss(single_cls_gt, single_cls_pred)
            single_l1_loss = self.smooth_l1(single_loc_gt, single_loc_pred, anchor_obj_idx)
            focal_loss.append(single_focal_loss)
            l1_loss.append(single_l1_loss)
        focal_loss, l1_loss = tf.reduce_mean(focal_loss), tf.reduce_mean(l1_loss)
        total_loss = focal_loss + l1_loss
        return total_loss, focal_loss, l1_loss

    def focal_loss(self, single_cls_gt, single_cls_pred, alpha=0.25, gamma=2.0):
        # single_cls_gt shape: [num_anchor], single_cls_pred shape: [num_anchor, num_class]
        num_classes = single_cls_pred.shape[-1]
        # gather the objects and background, discard the rest
        anchor_obj_idx = tf.where(tf.greater_equal(single_cls_gt, 0))
        anchor_obj_bg_idx = tf.where(tf.greater_equal(single_cls_gt, -1))
        anchor_obj_count = tf.cast(tf.shape(anchor_obj_idx)[0], tf.float32)
        single_cls_gt = tf.one_hot(single_cls_gt, num_classes)
        single_cls_gt = tf.gather_nd(single_cls_gt, anchor_obj_bg_idx)
        single_cls_pred = tf.gather_nd(single_cls_pred, anchor_obj_bg_idx)
        single_cls_gt = tf.reshape(single_cls_gt, (-1, 1))
        single_cls_pred = tf.reshape(single_cls_pred, (-1, 1))
        # compute the focal weight on each selected anchor box
        alpha_factor = tf.ones_like(single_cls_gt) * alpha
        alpha_factor = tf.where(tf.equal(single_cls_gt, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(tf.equal(single_cls_gt, 1), 1 - single_cls_pred, single_cls_pred)
        focal_weight = alpha_factor * focal_weight**gamma / anchor_obj_count
        cls_loss = tf.losses.BinaryCrossentropy(reduction='sum')(single_cls_gt,
                                                                 single_cls_pred,
                                                                 sample_weight=focal_weight)
        return cls_loss, anchor_obj_idx

    def smooth_l1(self, single_loc_gt, single_loc_pred, anchor_obj_idx, beta=0.1):
        # single_loc_gt shape: [num_anchor x 4], anchor_obj_idx shape:  [num_anchor x 4]
        single_loc_pred = tf.gather_nd(single_loc_pred, anchor_obj_idx)  # anchor_obj_count x 4
        single_loc_gt = tf.gather_nd(single_loc_gt, anchor_obj_idx)  # anchor_obj_count x 4
        anchor_obj_count = tf.cast(tf.shape(single_loc_pred)[0], tf.float32)
        single_loc_gt = tf.reshape(single_loc_gt, (-1, 1))
        single_loc_pred = tf.reshape(single_loc_pred, (-1, 1))
        loc_diff = tf.abs(single_loc_gt - single_loc_pred)
        cond = tf.less(loc_diff, beta)
        loc_loss = tf.where(cond, 0.5 * loc_diff**2 / beta, loc_diff - 0.5 * beta)
        loc_loss = tf.reduce_sum(loc_loss) / anchor_obj_count
        return loc_loss


class PredictBox(TensorOp):
    """Convert network output to bounding boxes.
    """
    def __init__(self,
                 inputs=None,
                 outputs=None,
                 mode=None,
                 input_shape=(512, 512, 3),
                 select_top_k=1000,
                 nms_max_outputs=100,
                 score_threshold=0.05):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.input_shape = input_shape
        self.select_top_k = select_top_k
        self.nms_max_outputs = nms_max_outputs
        self.score_threshold = score_threshold
        self.all_anchors, self.num_anchors_per_level = _get_fpn_anchor_box(width=input_shape[1], height=input_shape[0])

    def forward(self, data, state):
        cls_pred, loc_pred = data  # [Batch, #anchor, #num_classes], [Batch, #anchor, 4]
        batch_size = cls_pred.shape[0]
        labels_pred, scores_pred = tf.argmax(cls_pred, axis=-1), tf.reduce_max(cls_pred, axis=-1)
        # loc_pred -> loc_abs
        x1_abs = loc_pred[..., 0] * self.all_anchors[..., 2] + self.all_anchors[..., 0]
        y1_abs = loc_pred[..., 1] * self.all_anchors[..., 3] + self.all_anchors[..., 1]
        w_abs = tf.math.exp(loc_pred[..., 2]) * self.all_anchors[..., 2]
        h_abs = tf.math.exp(loc_pred[..., 3]) * self.all_anchors[..., 3]
        x2_abs, y2_abs = x1_abs + w_abs, y1_abs + h_abs
        # iterate over images
        final_results = []
        for idx in range(batch_size):
            scores_pred_single = scores_pred[idx]
            boxes_pred_single = tf.stack([y1_abs[idx], x1_abs[idx], y2_abs[idx], x2_abs[idx]], axis=-1)
            # iterate over each pyramid to select top 1000 anchor boxes
            start = 0
            top_idx = []
            for num_anchors_fpn_level in self.num_anchors_per_level:
                fpn_scores = scores_pred_single[start:start + num_anchors_fpn_level]
                selected_index = tf.math.top_k(fpn_scores, min(self.select_top_k, int(num_anchors_fpn_level))).indices
                top_idx.append(selected_index + start)
                start += num_anchors_fpn_level
            top_idx = tf.concat(top_idx, axis=0)
            # perform nms
            nms_keep = tf.image.non_max_suppression(tf.gather(boxes_pred_single, top_idx),
                                                    tf.gather(scores_pred_single, top_idx),
                                                    self.nms_max_outputs)
            top_idx = tf.gather(top_idx, nms_keep)  # narrow the keep index
            # mark the select as 0 for any anchorbox with score lower than threshold
            results_single = [
                tf.gather(x1_abs[idx], top_idx),
                tf.gather(y1_abs[idx], top_idx),
                tf.gather(w_abs[idx], top_idx),
                tf.gather(h_abs[idx], top_idx),
                tf.cast(tf.gather(labels_pred[idx], top_idx), tf.float32),
                tf.gather(scores_pred[idx], top_idx),
                tf.ones_like(tf.gather(x1_abs[idx], top_idx))
            ]
            # clip bounding boxes to image size
            results_single[0] = tf.clip_by_value(results_single[0],
                                                 clip_value_min=0,
                                                 clip_value_max=self.input_shape[1])
            results_single[1] = tf.clip_by_value(results_single[1],
                                                 clip_value_min=0,
                                                 clip_value_max=self.input_shape[0])
            results_single[2] = tf.clip_by_value(results_single[2],
                                                 clip_value_min=0,
                                                 clip_value_max=self.input_shape[1] - results_single[0])
            results_single[3] = tf.clip_by_value(results_single[3],
                                                 clip_value_min=0,
                                                 clip_value_max=self.input_shape[0] - results_single[1])
            # mark the select as 0 for any anchorbox with score lower than threshold
            results_single[-1] = tf.where(results_single[-2] > self.score_threshold,
                                          results_single[-1],
                                          tf.zeros_like(results_single[-1]))
            final_results.append(tf.stack(results_single, axis=-1))
        return tf.stack(final_results)


def exponential_decay(time, init_lr, gamma=0.9, start=1):
    return init_lr * gamma**(time - start)


def get_estimator(data_dir,
                  weights_path,
                  max_train_steps_per_epoch=1500,
                  batch_size=32,
                  epochs=30,
                  image_size=256,
                  num_classes=90,
                  control_frequency=3):
    # pipeline
    train_ds, eval_ds = mscoco.load_data(root_dir=data_dir)
    pipeline = fe.Pipeline(
        train_data=train_ds,
        eval_data=eval_ds,
        batch_size=batch_size,
        ops=[
            ReadImage(inputs="image", outputs="image"),
            LongestMaxSize(image_size,
                           image_in="image",
                           image_out="image",
                           bbox_in="bbox",
                           bbox_out="bbox",
                           bbox_params=BboxParams("coco", min_area=1.0)),
            PadIfNeeded(
                image_size,
                image_size,
                border_mode=cv2.BORDER_CONSTANT,
                image_in="image",
                image_out="image",
                bbox_in="bbox",
                bbox_out="bbox",
                bbox_params=BboxParams("coco", min_area=1.0),
            ),
            Sometimes(
                HorizontalFlip(mode="train",
                               image_in="image",
                               image_out="image",
                               bbox_in="bbox",
                               bbox_out="bbox",
                               bbox_params='coco')),
            Normalize(inputs="image", outputs="image", mean=1.0, std=1.0, max_pixel_value=127.5),
            ShiftLabel(inputs="bbox", outputs="bbox"),
            AnchorBox(inputs="bbox", outputs="anchorbox", width=image_size, height=image_size)
        ],
        pad_value=0)
    # network
    model = fe.build(model_fn=lambda: RetinaNet(input_shape=(image_size, image_size, 3), num_classes=num_classes),
                     optimizer_fn=lambda: tf.optimizers.SGD(0.001, momentum=0.9))
    controller = fe.build(model_fn=lstm_stacked, optimizer_fn=None, weights_path=weights_path)
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="image", outputs=["cls_pred", "loc_pred"]),
        RetinaLoss(inputs=["anchorbox", "cls_pred", "loc_pred"], outputs=["total_loss", "focal_loss", "l1_loss"]),
        UpdateOp(model=model, loss_name="total_loss"),
        PredictBox(
            input_shape=(image_size, image_size, 3), inputs=["cls_pred", "loc_pred"], outputs="pred", mode="eval")
    ])
    # estimator
    traces = [
        MeanAveragePrecision(num_classes=num_classes,
                             true_key='bbox',
                             pred_key='pred',
                             mode="eval"),
        LRController(model=model,
                     controller=controller,
                     control_frequency=control_frequency,
                     loss_key="total_loss"),
        RecordValLoss(control_frequency=control_frequency,
                      use_val_loss=True,
                      loss_key="total_loss")
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             max_train_steps_per_epoch=max_train_steps_per_epoch)
    return estimator
