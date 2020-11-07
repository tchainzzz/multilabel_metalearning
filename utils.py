import tensorflow as tf
from sklearn.metrics import precision_score, recall_score

# Loss utilities
def cross_entropy_loss(pred, label):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=tf.stop_gradient(label)))


def accuracy(labels, predictions):
    tf.print(labels, predictions)
    return tf.reduce_mean(tf.cast(tf.equal(labels, predictions), dtype=tf.float32))


def precision(labels, predictions, average='macro'):
    unique_preds, indices = tf.unique(predictions)
    class_prec = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    for i in range(len(unique_preds)):
        pred = unique_preds[i]
        prec = tf.reduce_mean(tf.cast(tf.equal(labels[predictions == pred], predictions[predictions==pred]), dtype=tf.float32))
        class_prec = class_prec.write(i, prec)
    return tf.reduce_mean(class_prec.stack())

@tf.function
def recall(labels, predictions, average='macro'):
    unique_labels, indices = tf.unique(labels)
    class_rec = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    for i in range(len(unique_labels)):
        lbl = unique_labels[i]
        rec = tf.reduce_mean(tf.cast(tf.equal(labels[labels == lbl], predictions[labels == lbl]), dtype=tf.float32))
        class_rec = class_rec.write(i, rec)

    return tf.reduce_mean(class_rec.stack())

@tf.function
def fscore(labels, predictions, beta=1):
    unique_preds, _ = tf.unique(predictions)
    unique_labels, _ = tf.unique(labels)
    unique_tokens, _ = tf.unique(tf.concat([unique_preds, unique_labels], axis=0))
    class_f = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    for i in range(len(unique_tokens)):
        curr_class = unique_tokens[i]
        pred_mask = tf.cast(tf.equal(labels[predictions == curr_class], predictions[predictions == curr_class]), dtype=tf.float32)
        label_mask = tf.cast(tf.equal(labels[labels == curr_class], predictions[labels == curr_class]), dtype=tf.float32)
        prec = tf.cond(tf.equal(tf.size(pred_mask), 0), lambda: tf.constant(0.0), lambda: tf.reduce_mean(pred_mask))
        rec = tf.cond(tf.equal(tf.size(label_mask), 0), lambda: tf.constant(0.0), lambda: tf.reduce_mean(label_mask))
        fscore = tf.cond(tf.logical_and(tf.equal(prec, 0), tf.equal(rec, 0)), lambda: tf.constant(0.0), lambda: (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec))
        class_f = class_f.write(i, fscore)
    return tf.reduce_mean(class_f.stack())

def convert_to_powerset(y):
    single_labels = (np.packbits(y.astype(int), 2, 'little') - 1).reshape((len(y), -1))
    one_hot = np.eye(num_classes)[single_labels]
    return one_hot

def support_query_split(X, y, converter, support_dim=1):
    X_tr, X_ts = tf.split(X, 2, axis=support_dim)
    y_new = converter(y)
    y_tr, y_ts = tf.split(y_new, 2, axis=support_dim)
    return X_tr, X_ts, y_tr, y_ts
