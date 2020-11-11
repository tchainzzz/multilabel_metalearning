import load_data_tf as load_data

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import os
import glob

tf.keras.backend.set_floatx('float64')
class ProtoNet(tf.keras.Model):

    def __init__(self, num_filters, latent_dim):
        super(ProtoNet, self).__init__()
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        num_filter_list = self.num_filters + [latent_dim]
        self.convs = []
        for i, num_filter in enumerate(num_filter_list):
            block_parts = [layers.Conv2D(
                filters=num_filter, kernel_size=3, padding='SAME', activation='linear'), ]

        block_parts += [layers.BatchNormalization()]
        block_parts += [layers.Activation('relu')]
        block_parts += [layers.MaxPool2D()]
        block = tf.keras.Sequential(block_parts, name='conv_block_%d' % i)
        self.__setattr__("conv%d" % i, block)
        self.convs.append(block)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inp):
        out = inp
        for conv in self.convs:
            out = conv(out)
        out = self.flatten(out)
        return out


def protoloss(x_latent, q_latent, labels_onehot, num_classes, num_support, num_queries):
    # return the cross-entropy loss and accuracy
    tf.random.shuffle(q_latent)
    centroids = tf.reduce_mean(tf.reshape(
        x_latent, (num_classes, num_support, -1)), 1)
    # reduce along "classes" dim to get per-class means. output [N, D]

    # procedure: transform centroids to [n*q, n, d] via a nqx repeat op
    centroids_repeat = tf.repeat(tf.expand_dims(
        centroids, 0), repeats=num_classes * num_queries, axis=0)
    # compute the prototypes
    # output [n*q, n, d] -- don't tempt the broadcasting gods
    q_latent = tf.repeat(tf.expand_dims(q_latent, 1),
                         repeats=num_classes, axis=1)
    # Sq. L2 for each (n*q, n) slice
    diffs = tf.reduce_sum(
        tf.square(tf.subtract(q_latent, centroids_repeat)), axis=-1)
    labels_reshaped = tf.reshape(
        labels_onehot, (num_classes * num_queries, num_classes))
    ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        tf.stop_gradient(labels_reshaped), -diffs))
    preds = tf.argmax(-diffs, axis=-1)
    labels = tf.argmax(labels_reshaped, axis=-1)
    acc = accuracy(preds, labels)
    return ce_loss, acc


def proto_net_train_step(model, optim, x, q, labels_ph):
    num_classes, num_support, im_height, im_width, channels = x.shape
    num_queries = q.shape[1]
    x = tf.reshape(x, [-1, im_height, im_width, channels])
    q = tf.reshape(q, [-1, im_height, im_width, channels])

    with tf.GradientTape() as tape:
        x_latent = model(x)
        q_latent = model(q)
        ce_loss, acc = ProtoLoss(x_latent, q_latent, labels_ph, num_classes, num_support, num_queries)

    gradients = tape.gradient(ce_loss, model.trainable_variables)
    optim.apply_gradients(zip(gradients, model.trainable_variables))
    return ce_loss, acc


def proto_net_eval(model, x, q, labels_ph):
    num_classes, num_support, im_height, im_width, channels = x.shape
    num_queries = q.shape[1]
    x = tf.reshape(x, [-1, im_height, im_width, channels])
    q = tf.reshape(q, [-1, im_height, im_width, channels])

    x_latent = model(x)
    q_latent = model(q)
    ce_loss, acc = ProtoLoss(
        x_latent, q_latent, labels_ph, num_classes, num_support, num_queries)

    return ce_loss, acc


def run_protonet(data_path='./omniglot_resized', n_way=3, k_shot=8, n_query=5):
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

    # call DataGenerator with k_shot+n_query samples per class
    meta_dataset = load_data.MetaBigEarthNetTaskDataset(data_dir=data_dir, support_size=k_shot+n_query, label_subset_size=n_way, split_save_path="smallearthnet.pkl", split_file="smallearthnet.pkl", data_format='channels_last')

    for ep in range(n_epochs):
        for epi in range(n_episodes):
            X, y, y_debug = meta_dataset.sample_batch(meta_batch_size=1, split='train')
            y = convert_to_powerset(y)
            _, s, h, w, c = X.shape
            X = tf.squeeze(X, axis=0)
            support, query = X[:k_shot, ...], X[k_shot:, ...]
            labels = tf.squeeze(y[:, :, :k_shot, :], 0)
            ls, ac = proto_net_train_step(model, optimizer, x=support, q=query, labels_ph=labels)
        if (epi+1) % 50 == 0:
            images, labels = meta_dataset.sample_batch(meta_batch_size=1, split='val')
            support, query = tf.reshape(images[:, :, :k_shot, :], (n_way, k_shot, im_height, im_width, channels)), tf.reshape(
                images[:, :, k_shot:, :], (n_way, n_query, im_height, im_width, channels))
            labels = tf.squeeze(labels[:, :, k_shot:, :], 0)
            val_ls, val_ac = proto_net_eval(
                model, x=support, q=query, labels_ph=labels)
            print('[epoch {}/{}, episode {}/{}] => meta-training loss: {:.5f}, meta-training acc: {:.5f}, meta-val loss: {:.5f}, meta-val acc: {:.5f}'.format(
                ep+1, n_epochs, epi+1, n_episodes, ls.numpy(), ac.numpy(), val_ls.numpy(), val_ac.numpy()))
        if (epi + 1) % 100 == 0:
            train_losses.append(ls.numpy())
            train_accs.append(ac.numpy())
            val_losses.append(val_ls.numpy())
            val_accs.append(val_ac.numpy())
    print('Testing...')
    meta_test_accuracies = []
    for epi in range(n_meta_test_episodes):
        images, labels = data_generator.sample_batch(
            'meta_test', 1, shuffle=False)
        support, query = tf.reshape(images[:, :, :k_meta_test_shot, :], (n_meta_test_way, k_meta_test_shot, im_height, im_width, channels)), tf.reshape(
            images[:, :, k_meta_test_shot:, :], (n_meta_test_way, n_meta_test_query, im_height, im_width, channels))
        labels = tf.squeeze(labels[:, :, k_meta_test_shot:, :], 0)
        ls, ac = proto_net_eval(model, x=support, q=query, labels_ph=labels)
        meta_test_accuracies.append(ac)
        if (epi+1) % 50 == 0:
            print('[meta-test episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(epi +
                                                                                  1, n_meta_test_episodes, ls, ac))
    avg_acc = np.mean(meta_test_accuracies)
    stds = np.std(meta_test_accuracies)
    print(
        'Average Meta-Test Accuracy: {:.5f}, Meta-Test Accuracy Std: {:.5f}'.format(avg_acc, stds))
    return train_losses, train_accs, val_losses, val_accs, avg_acc, stds

from options import *
if __name__ == '__main__':
    args = get_args()
    results = run_protonet(args.data_dir, n_way=n, k_shot=k, n_query=q, n_meta_test_way=test_n, k_meta_test_shot=test_k, n_meta_test_query=test_q)
