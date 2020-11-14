import tensorflow as tf
from sklearn.metrics import precision_score, recall_score
import numpy as np
import datetime
import pytz

# magic numbers obtained from running `python3 calculate_mean.py`
CHANNEL_MEANS = [0.19261545, 0.24894128, 0.1618804]
CHANNEL_STDS = [0.17641555, 0.14091561, 0.11086669]

# Loss utilities
def cross_entropy_loss(pred, label):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=tf.stop_gradient(label)))


def accuracy(labels, predictions):
    return tf.reduce_mean(tf.cast(tf.equal(labels, predictions), dtype=tf.float32))

@tf.function
def lazy_onehot(labels, predictions, n_classes):
    n_classes = tf.cast(n_classes, labels.dtype)
    labels = tf.math.floormod(tf.bitwise.right_shift(tf.expand_dims(labels, 1), tf.range(n_classes)), 2)
    predictions = tf.math.floormod(tf.bitwise.right_shift(tf.expand_dims(predictions, 1), tf.range(n_classes)), 2)
    return labels, predictions

@tf.function
def precision(labels, predictions, n_classes, multi='powerset'):
    class_prec = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    if multi == 'old_powerset':
        unique_preds, indices = tf.unique(predictions)
        for i in range(len(unique_preds)):
            pred = unique_preds[i]
            prec = tf.reduce_mean(tf.cast(tf.equal(labels[predictions == pred], predictions[predictions==pred]), dtype=tf.float32))
            class_prec = class_prec.write(i, prec)
    elif multi == 'binary' or multi == 'powerset':
        if multi == 'powerset':
            labels, predictions = lazy_onehot(labels, predictions, n_classes)
        _, n_classes = labels.shape
        for i in range(n_classes):
            class_labels = labels[:, i]
            class_predictions = predictions[:, i]
            prec_mask = tf.cast(tf.equal(class_labels[class_predictions == 1], class_predictions[class_predictions == 1]), dtype=tf.float32)
            prec = tf.cond(tf.equal(tf.size(prec_mask), 0), lambda: tf.constant(0.0), lambda: tf.reduce_mean(prec_mask))
            class_prec = class_prec.write(i, prec)
    else:
        raise NotImplementedError()
    return tf.reduce_mean(class_prec.stack())

@tf.function
def recall(labels, predictions, n_classes, multi='powerset'):
    class_rec = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    if multi == 'old_powerset':
        unique_labels, indices = tf.unique(labels)
        for i in range(len(unique_labels)):
            lbl = unique_labels[i]
            rec = tf.reduce_mean(tf.cast(tf.equal(labels[labels == lbl], predictions[labels == lbl]), dtype=tf.float32))
            class_rec = class_rec.write(i, rec)
    elif multi == 'binary' or multi == 'powerset':
        if multi == 'powerset':
            labels, predictions = lazy_onehot(labels, predictions, n_classes)

        _, n_classes = labels.shape
        for i in range(n_classes):
            class_labels = labels[:, i]
            class_predictions = predictions[:, i]

            rec_mask = tf.cast(tf.equal(class_labels[class_labels == 1], class_predictions[class_labels == 1]), dtype=tf.float32)
            rec = tf.cond(tf.equal(tf.size(rec_mask), 0), lambda: tf.constant(0.0), lambda: tf.reduce_mean(rec_mask))
            class_rec = class_rec.write(i, rec)
    else:
        raise NotImplementedError()
    return tf.reduce_mean(class_rec.stack())

@tf.function
def fscore(labels, predictions, n_classes, multi='powerset', beta=1):
    class_f = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    if multi == 'old_powerset':
        unique_preds, _ = tf.unique(predictions)
        unique_labels, _ = tf.unique(labels)
        unique_tokens, _ = tf.unique(tf.concat([unique_preds, unique_labels], axis=0))
        for i in range(len(unique_tokens)):
            curr_class = unique_tokens[i]
            pred_mask = tf.cast(tf.equal(labels[predictions == curr_class], predictions[predictions == curr_class]), dtype=tf.float32)
            label_mask = tf.cast(tf.equal(labels[labels == curr_class], predictions[labels == curr_class]), dtype=tf.float32)
            prec = tf.cond(tf.equal(tf.size(pred_mask), 0), lambda: tf.constant(0.0), lambda: tf.reduce_mean(pred_mask))
            rec = tf.cond(tf.equal(tf.size(label_mask), 0), lambda: tf.constant(0.0), lambda: tf.reduce_mean(label_mask))
            fscore = tf.cond(tf.logical_and(tf.equal(prec, 0), tf.equal(rec, 0)), lambda: tf.constant(0.0), lambda: (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec))
            class_f = class_f.write(i, fscore)
    elif multi == 'binary' or multi == 'powerset':
        if multi == 'powerset':
            labels, predictions = lazy_onehot(labels, predictions, n_classes)
        _, n_classes = labels.shape
        for i in range(n_classes):
            class_labels = labels[:, i]
            class_predictions = predictions[:, i]
            prec_mask = tf.cast(tf.equal(class_labels[class_predictions == 1], class_predictions[class_predictions == 1]), dtype=tf.float32)
            prec = tf.cond(tf.equal(tf.size(prec_mask), 0), lambda: tf.constant(0.0), lambda: tf.reduce_mean(prec_mask))
            rec_mask = tf.cast(tf.equal(class_labels[class_labels == 1], class_predictions[class_labels == 1]), dtype=tf.float32)
            rec = tf.cond(tf.equal(tf.size(rec_mask), 0), lambda: tf.constant(0.0), lambda: tf.reduce_mean(rec_mask))
            fscore = tf.cond(tf.logical_and(tf.equal(prec, 0), tf.equal(rec, 0)), lambda: tf.constant(0.0), lambda: (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec))
            class_f = class_f.write(i, fscore)
    else:
        raise NotImplementedError()
    return tf.reduce_mean(class_f.stack())

def convert_to_powerset(y):
    _, _, subset_size = y.shape # (batch_size, support_size, label_subset_size)
    num_classes = (1 << subset_size) - 1
    single_labels = (np.packbits(y.astype(int), 2, 'little') - 1).reshape((len(y), -1))
    one_hot = np.eye(num_classes)[single_labels]
    return one_hot.astype(np.float32)

def convert_to_bin_rel(y):
    return tf.stack([1-y, y], axis = -1)


def support_query_split(X, y, converter, support_dim=1):
    X_tr, X_ts = tf.split(X, 2, axis=support_dim)
    y_new = converter(y)
    y_tr, y_ts = tf.split(y_new, 2, axis=support_dim)
    return X_tr, X_ts, y_tr, y_ts

def generate_experiment_name(experiment_name, extra_tokens=[], timestamp=True):
    experiment_tokens = []
    if timestamp:
        utc_now = pytz.utc.localize(datetime.datetime.utcnow())
        pst_now = utc_now.astimezone(pytz.timezone("America/Los_Angeles"))
        current_time = pst_now.strftime("%Y-%m-%d-%H:%M:%S")
        experiment_tokens.append(current_time)
    if experiment_name: 
        experiment_tokens.append(experiment_name)
    if len(extra_tokens):
        experiment_tokens.extend(extra_tokens)
    experiment_fullname = "_".join(experiment_tokens)
    return experiment_fullname

def normalize(img,data_format='channels_last'):
    if data_format == 'channels_last':
        means = CHANNEL_MEANS
        stds = CHANNEL_STDS
    elif data_format == 'channels_first':
        means = CHANNEL_MEANS[:, np.newaxis, np.newaxis]
        stds = CHANNEL_MEANS[:, np.newaxis, np.newaxis]
    return (img - CHANNEL_MEANS) / (CHANNEL_STDS)

