import load_data_tf as load_data

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import os
import glob

from utils import *

class ProtoNet(tf.keras.Model):

    def __init__(self, num_filters, latent_dim):
        super(ProtoNet, self).__init__()
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        num_filter_list = self.num_filters + [latent_dim]
        self.convs = []
        for i, num_filter in enumerate(num_filter_list):
            block_parts = [layers.Conv2D(filters=num_filter, kernel_size=3, padding='SAME', activation='linear'), ]
            block_parts += [layers.BatchNormalization()]
            block_parts += [layers.Activation('relu')]
            block_parts += [layers.MaxPool2D()]
            block = tf.keras.Sequential(block_parts, name='conv_block_%d' % i)
            self.__setattr__("conv%d" % i, block)
            self.convs.append(block)
        self.flatten = tf.keras.layers.Flatten()
        self.embed = tf.keras.layers.Dense(latent_dim)

    def call(self, inp):
        out = inp
        for conv in self.convs:
            out = conv(out)
        out = self.flatten(out)
        out = self.embed(out)
        return out


def ProtoLoss(x_latent, q_latent, labels_onehot, num_classes, num_support, num_queries):
    # return the cross-entropy loss and accuracy
    tf.random.shuffle(q_latent)
    labels_categorical = tf.argmax(labels_onehot, axis=-1)
    # x_latent has shape (support, embed_dim)
    centroids = []
    embed_size = x_latent.shape[-1] 
    for i in range(num_classes):
        support_class = x_latent[labels_categorical == i]
        centroid = tf.cond(tf.equal(tf.size(support_class), 0), lambda: tf.constant([0.] * embed_size), lambda: tf.reduce_mean(support_class, axis=0))
        centroids.append(centroid)
    centroids = tf.stack(centroids, axis=0)
    # reduce along "classes" dim to get per-class means. output [N, D]

    # procedure: transform centroids to [n*q, n, d] via a nqx repeat op
    centroids_repeat = tf.repeat(tf.expand_dims(centroids, 0), repeats=num_queries, axis=0) # from (n_classes, embed_dim) -> (n_query, n_classes, embed_dim)

    # compute the prototypes
    q_latent = tf.repeat(tf.expand_dims(q_latent, 1), repeats=num_classes, axis=1) # parallel expansion from (n_query, embed_dim) -> (n_query, n_classes, embed_dim)
    # Sq. L2 for each (n*q, n) slice

    diffs = tf.reduce_sum(tf.square(tf.subtract(q_latent, centroids_repeat)), axis=-1) 
    # shape should be (n_query, n_classes)

    #diffs = tf.clip_by_value(diffs, tf.reduce_min(diffs), tf.reduce_max(diffs))
    n_query = q_latent.shape[0]
    mask = tf.repeat(tf.clip_by_value(tf.reduce_sum(labels_onehot, axis=0, keepdims=True), 0, 1), repeats=n_query, axis=0)
    replace = tf.ones_like(diffs) * tf.reduce_max(diffs)
    diffs = diffs * mask + replace * (1 - mask)
    ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.stop_gradient(labels_onehot), -diffs))
    preds = tf.argmax(-diffs, axis=-1)
    labels = tf.argmax(labels_onehot, axis=-1)
    acc = accuracy(preds, labels)
    return ce_loss, acc


def proto_net_train_step(model, optim, x, q, labels_ph):
    num_support, im_height, im_width, channels = x.shape
    num_queries = q.shape[0]
    num_classes = labels_ph.shape[1]
    #x = tf.reshape(x, [-1, im_height, im_width, channels])
    #q = tf.reshape(q, [-1, im_height, im_width, channels])

    with tf.GradientTape() as tape:
        x_latent = model(x)
        q_latent = model(q)
        ce_loss, acc = ProtoLoss(x_latent, q_latent, labels_ph, num_classes, num_support, num_queries)
    gradients = tape.gradient(ce_loss, model.trainable_variables)
    optim.apply_gradients(zip(gradients, model.trainable_variables))
    return ce_loss, acc


def proto_net_eval(model, x, q, labels_ph):

    num_support, im_height, im_width, channels = x.shape
    num_queries = q.shape[0]
    num_classes = labels_ph.shape[1]
    #x = tf.reshape(x, [-1, im_height, im_width, channels])
    #q = tf.reshape(q, [-1, im_height, im_width, channels])

    x_latent = model(x)
    q_latent = model(q)
    ce_loss, acc = ProtoLoss(x_latent, q_latent, labels_ph, num_classes, num_support, num_queries)

    return ce_loss, acc


def run_protonet(data_root='../cs330-storage', n_way=3, n_support=8, n_query=8, n_meta_test_way=3, n_meta_test_support=8, n_meta_test_query=8):
    n_epochs = 20
    n_episodes = 100

    num_filters = 32
    latent_dim = 16
    num_conv_layers = 3
    n_meta_test_episodes = 1000

    model = ProtoNet([num_filters] * num_conv_layers, latent_dim)
    optimizer = tf.keras.optimizers.Adam()

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    filter_files = [os.path.join(data_root, 'patches_with_cloud_and_shadow.csv'), os.path.join(data_root, 'patches_with_seasonal_snow.csv')]  # replace with your path
    data_dir = os.path.join(data_root, "SmallEarthNet")
    meta_dataset = load_data.MetaBigEarthNetTaskDataset(data_dir=data_dir, support_size=n_support+n_query, label_subset_size=n_way, split_save_path="smallearthnet.pkl", split_file="smallearthnet.pkl", data_format='channels_last')

    for ep in range(n_epochs):
        for epi in range(n_episodes):
            X, y, y_debug = meta_dataset.sample_batch(batch_size=1, split='train', mode='permutation')
            y = convert_to_powerset(y)
            # Powerset shapes: (1, support + query, h, w, c); (1, support + query, 2^n_classes - 1)
            # BR shapes: " , (1, support + query, n_classes, 2)
            _, s, h, w, c = X.shape
            X = tf.squeeze(X, axis=0)
            support, query = X[:n_support, ...], X[n_support:, ...]
            labels = tf.squeeze(y[:, n_support:, ...], 0)
            ls, ac = proto_net_train_step(model, optimizer, x=support, q=query, labels_ph=labels)
        X, y, y_debug = meta_dataset.sample_batch(batch_size=1, split='val', mode='permutation')
        X = tf.squeeze(X, axis=0)
        support, query = X[:n_support, ...], X[:n_support, ...]
        labels = tf.squeeze(y[:, n_support:, ...], 0)
        val_ls, val_ac = proto_net_eval(model, x=support, q=query, labels_ph=labels)
        print('[epoch {}/{}, episode {}/{}] => meta-training loss: {:.5f}, meta-training acc: {:.5f}, meta-val loss: {:.5f}, meta-val acc: {:.5f}'.format(ep+1, n_epochs, epi+1, n_episodes, ls.numpy(), ac.numpy(), val_ls.numpy(), val_ac.numpy()))
    if (epi + 1) % 100 == 0:
        train_losses.append(ls.numpy())
        train_accs.append(ac.numpy())
        val_losses.append(val_ls.numpy())
        val_accs.append(val_ac.numpy())
    print('Testing...')
    meta_test_accuracies = []
    for epi in range(n_meta_test_episodes):
        X, y, y_debug = meta_dataset.sample_batch(Batch_size=1, split='test', mode='permutation')
        support, query = X[:n_support, ...], X[n_support, ...]
        labels = tf.squeeze(labels[:, :, n_meta_test_support:, :], 0)
        ls, ac = proto_net_eval(model, x=support, q=query, labels_ph=labels)
        meta_test_accuracies.append(ac)
        if (epi+1) % 50 == 0:
            print('[meta-test episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(epi + 1, n_meta_test_episodes, ls, ac))
    avg_acc = np.mean(meta_test_accuracies)
    stds = np.std(meta_test_accuracies)
    print(
        'Average Meta-Test Accuracy: {:.5f}, Meta-Test Accuracy Std: {:.5f}'.format(avg_acc, stds))
    return train_losses, train_accs, val_losses, val_accs, avg_acc, stds

from options import *
if __name__ == '__main__':
    args = get_args()
    results = run_protonet(args.data_root, n_way=args.label_subset_size, n_support=args.support_size, n_query=args.support_size, n_meta_test_way=args.label_subset_size, n_meta_test_support=args.support_size, n_meta_test_query=args.support_size)
