import load_data_tf as load_data

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import os
import glob

from pathlib import Path
from utils import *
from tensorboardX import SummaryWriter

from mann import SNAILConvBlock
import time

class ProtoNet(tf.keras.Model):

    def __init__(self, num_filters, latent_dim, num_classes, num_blocks=4, multi='powerset'):
        super(ProtoNet, self).__init__()
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.multi = multi
        self.latent_dim = latent_dim
        #num_filter_list = self.num_filters + [latent_dim]
        #self.convs = []
        """for i, num_filter in enumerate(num_filter_list):
            block_parts = [layers.Conv2D(filters=num_filter, kernel_size=3, padding='SAME', activation='linear'), ]
            block_parts += [layers.BatchNormalization()]
            block_parts += [layers.Activation('relu')]
            block_parts += [layers.MaxPool2D()]
            block = tf.keras.Sequential(block_parts, name='conv_block_%d' % i)
            self.__setattr__("conv%d" % i, block)
            self.convs.append(block)"""

        self.convs = [SNAILConvBlock(num_filters) for _ in range(num_blocks)]
        self.flatten = tf.keras.layers.Flatten()
        #if multi == 'powerset':
        #    self.embed = tf.keras.layers.Dense(latent_dim)
        #else:
        #    self.embed = [tf.keras.layers.Dense(latent_dim) for _ in range(num_classes)]

    def call(self, inp):
        out = inp
        for conv in self.convs:
            out = conv(out)
        out = self.flatten(out)
        """
        if self.multi == 'powerset':
            out = self.embed(out)
        else:
            out = tf.stack([layer(out) for layer in self.embed], axis=1)"""
        return out

    def safe_masked_centroid(self, support, embed_size, placeholder=0.):
        return tf.cond(tf.equal(tf.size(support), 0), lambda: tf.constant([placeholder] * embed_size), lambda: tf.reduce_mean(support, axis=0))

    def ProtoLoss(self, x_latent, q_latent, labels_onehot, num_classes, num_support, num_queries, label_subset_size):
        # return the cross-entropy loss and accuracy
        tf.random.shuffle(q_latent)
        labels_categorical = tf.argmax(labels_onehot, axis=-1)
        # x_latent has shape (support, embed_dim)
        centroids = []
        embed_size = x_latent.shape[-1] 
        for i in range(num_classes):
            # select all examples (i.e. "rows" of x_latent) relevant to centroid
            if self.multi == 'powerset':
                support_class = x_latent[labels_categorical == i]
                """
                centroid is:
                    computed as normal if class support is non-zero (i.e. more than 1 example
                    from class)
                    set to some arbitrary constant if class support is zero (avg. of zero-length
                    slice is undefined), then masked out
                """
                centroid = self.safe_masked_centroid(support_class, embed_size) 
            else:
                bin_rel_pos_mask = (labels_onehot[:, i, 1] == 1)
                bin_rel_neg_mask = (labels_onehot[:, i, 0] == 1)
                support_class, not_support_class = x_latent[:, i, :][bin_rel_pos_mask], x_latent[:, i, :][bin_rel_neg_mask]
                pos_centroid, neg_centroid = self.safe_masked_centroid(support_class, embed_size), self.safe_masked_centroid(not_support_class, embed_size)
                centroid = tf.stack([pos_centroid, neg_centroid], axis=0)
            centroids.append(centroid)
        centroids = tf.stack(centroids, axis=0)
        # reduce along "classes" dim to get per-class means. output [N, D]

        # procedure: transform centroids to [n*q, n, d] via a nqx repeat op
        centroids_repeat = tf.repeat(tf.expand_dims(centroids, 0), repeats=num_queries, axis=0) # from (n_classes, embed_dim) -> (n_query, n_classes, embed_dim)

        # compute the prototypes
        if self.multi == 'powerset':
            q_latent = tf.repeat(tf.expand_dims(q_latent, 1), repeats=num_classes, axis=1) # parallel expansion from (n_query, embed_dim) -> (n_query, n_classes, embed_dim)
        else:
            q_latent = tf.repeat(tf.expand_dims(q_latent, 2), repeats=2, axis=2)

        """
            else if multi == binary, we already have per-class OVR embeddings, 
        """

        diffs = tf.reduce_sum(tf.square(tf.subtract(q_latent, centroids_repeat)), axis=-1) 
        # shape should be (n_query, n_classes) and 2 if binary relevance

        #diffs = tf.clip_by_value(diffs, tf.reduce_min(diffs), tf.reduce_max(diffs))
        n_query = q_latent.shape[0]

        pre_mask_onehot = labels_onehot if self.multi == 'powerset' else labels_onehot[..., 1]
        mask = tf.repeat(tf.clip_by_value(tf.reduce_sum(pre_mask_onehot, axis=0, keepdims=True), 0, 1), repeats=n_query, axis=0) # columns of 0s (class does not appear in query) and 1s (class does appear in query)

        replace = tf.ones_like(diffs) * tf.reduce_max(diffs) # masked out w/ max (i.e. replace non well-defined centroids w/ min. energy logit output
        if self.multi == 'binary': mask = tf.repeat(tf.expand_dims(mask, 2), repeats=2, axis=2)
        diffs = diffs * mask + replace * (1 - mask) # 0-1 mask -- can't assign to eager tensor
        ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.stop_gradient(labels_onehot), -diffs))
        preds = tf.argmax(-diffs, axis=-1)
        labels = tf.argmax(labels_onehot, axis=-1)
        prec = precision(preds, labels, label_subset_size, self.multi)
        rec = recall(preds, labels, label_subset_size, self.multi)
        f1 = fscore(preds, labels, label_subset_size, self.multi)
        return ce_loss, prec, rec, f1


def proto_net_train_step(model, optim, x, q, labels_ph, label_subset_size):
    num_support, im_height, im_width, channels = x.shape
    num_queries = q.shape[0]
    num_classes = labels_ph.shape[1]
    #x = tf.reshape(x, [-1, im_height, im_width, channels])
    #q = tf.reshape(q, [-1, im_height, im_width, channels])

    with tf.GradientTape() as tape:
        x_latent = model(x)
        q_latent = model(q)
        ce_loss, prec, rec, f1 = model.ProtoLoss(x_latent, q_latent, labels_ph, num_classes, num_support, num_queries, label_subset_size)
    gradients = tape.gradient(ce_loss, model.trainable_variables)
    optim.apply_gradients(zip(gradients, model.trainable_variables))
    return ce_loss, prec, rec, f1


def proto_net_eval(model, x, q, labels_ph, label_subset_size):

    num_support, im_height, im_width, channels = x.shape
    num_queries = q.shape[0]
    num_classes = labels_ph.shape[1]
    #x = tf.reshape(x, [-1, im_height, im_width, channels])
    #q = tf.reshape(q, [-1, im_height, im_width, channels])

    x_latent = model(x)
    q_latent = model(q)
    ce_loss, prec, rec, f1 = model.ProtoLoss(x_latent, q_latent, labels_ph, num_classes, num_support, num_queries, label_subset_size)

    return ce_loss, prec, rec, f1


def run_protonet(data_root='../cs330-storage', n_way=3, n_support=8, n_query=8, n_meta_test_way=3, n_meta_test_support=8, n_meta_test_query=8, multi='powerset', experiment_name=None, n_episodes=10000, latent_dim=16, lr=1e-3, num_filters=64, log_frequency=5, patience=200):

    n_meta_test_episodes = 1000
    experiment_fullname = generate_experiment_name(experiment_name, extra_tokens=[Path(__file__).stem])

    log_dir = '../tensorboard_logs/' + experiment_fullname
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    model = ProtoNet(num_filters, latent_dim, n_way, multi=multi)
    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=2000,
        decay_rate=0.5,
        staircase=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    filter_files = [os.path.join(data_root, 'patches_with_cloud_and_shadow.csv'), os.path.join(data_root, 'patches_with_seasonal_snow.csv')]  # replace with your path
    data_dir = os.path.join(data_root, "SmallEarthNet")
    meta_dataset = load_data.MetaBigEarthNetTaskDataset(data_dir=data_dir, support_size=n_support+n_query, label_subset_size=n_way, filter_files=filter_files, split_save_path="smallearthnet.pkl", split_file="smallearthnet.pkl", data_format='channels_last')

    best_episode, best_val_loss = 0, float('inf')
    for ep in range(n_episodes):
        start = time.time()
        ls, prec, rec, f1 = [], [], [], []
        X, y, y_debug = meta_dataset.sample_batch(batch_size=1, split='train', mode='permutation')
        y = convert_to_powerset(y) if multi == 'powerset' else convert_to_bin_rel(y)
        # Powerset shapes: (1, support + query, h, w, c); (1, support + query, 2^n_classes - 1)
        # BR shapes: " , (1, support + query, n_classes, 2)
        _, s, h, w, c = X.shape
        X = tf.squeeze(X, axis=0)
        support, query = X[:n_support, ...], X[n_support:, ...]
        labels = tf.squeeze(y[:, n_support:, ...], 0)
        ls_tr, prec_tr, rec_tr, f1_tr = proto_net_train_step(model, optimizer, x=support, q=query, labels_ph=labels, label_subset_size=n_way)
        ls.append(ls_tr.numpy())
        prec.append(prec_tr.numpy())
        rec.append(rec_tr.numpy())
        f1.append(f1_tr.numpy())
        mean_ls = np.mean(ls)
        mean_prec = np.mean(prec)
        mean_rec = np.mean(rec)
        mean_f1 = np.mean(f1)
        writer.add_scalar('Meta-train loss', mean_ls, ep)
        writer.add_scalar('Meta-train precision', mean_prec, ep)
        writer.add_scalar('Meta-train recall', mean_rec, ep)
        writer.add_scalar('Meta-train F1', mean_f1, ep)
        X, y, y_debug = meta_dataset.sample_batch(batch_size=1, split='val', mode='permutation')
        y = convert_to_powerset(y) if multi == 'powerset' else convert_to_bin_rel(y)

        X = tf.squeeze(X, axis=0)
        support, query = X[:n_support, ...], X[n_support:, ...]
        labels = tf.squeeze(y[:, n_support:, ...], 0)
        val_ls, val_prec, val_rec, val_f1 = proto_net_eval(model, x=support, q=query, labels_ph=labels, label_subset_size=n_way)


        writer.add_scalar('Meta-validation loss', val_ls.numpy(), ep)
        writer.add_scalar('Meta-validation precision', val_prec.numpy(), ep)
        writer.add_scalar('Meta-validation recall', val_rec.numpy(), ep)
        writer.add_scalar('Meta-validation F1', val_f1.numpy(), ep)

        if (ep + 1) % log_frequency == 0:
            print('Iteration {}/{} - meta-train loss/prec/rec/f1: {:.4f}/{:.4f}/{:.4f}/{:.4f}, meta-val loss/prec/rec/f1: {:.4f}/{:.4f}/{:.4f}/{:.4f}, took {:.4f}s'.format(ep+1, n_episodes, mean_ls, mean_prec, mean_rec, mean_f1, val_ls.numpy(), val_prec.numpy(), val_rec.numpy(), val_f1.numpy(), time.time() - start))
    #if (epi + 1) % 100 == 0:
    #    train_losses.append(ls.numpy())
    #    train_accs.append(ac.numpy())
    #    val_losses.append(val_ls.numpy())
    #    val_accs.append(val_ac.numpy())

        if val_ls.numpy() < best_val_loss:
            best_episode = ep
            best_val_loss = val_ls.numpy()
            print("Validation loss improved at epoch", ep+1)
        if best_episode + patience <= ep:
            print("Validation loss failed to improve for {} epochs. Stopping at epoch {}/{}".format(patience, ep+1, n_episodes))
            break
    print('Testing...')
    meta_test_loss, meta_test_prec, meta_test_rec, meta_test_f1 = [], [], [], []
    for epi in range(n_meta_test_episodes):
        X, y, y_debug = meta_dataset.sample_batch(batch_size=1, split='test', mode='permutation')
        X = tf.squeeze(X, axis=0)
        y = convert_to_powerset(y) if multi == 'powerset' else convert_to_bin_rel(y)

        support, query = X[:n_support, ...], X[n_support:, ...]
        labels = tf.squeeze(y[:, n_meta_test_support:, :], 0)
        ls_ts, prec_ts, rec_ts, f1_ts = proto_net_eval(model, x=support, q=query, labels_ph=labels, label_subset_size=n_meta_test_way)
        meta_test_loss.append(ls_ts)
        meta_test_prec.append(prec_ts)
        meta_test_rec.append(rec_ts)
        meta_test_f1.append(f1_ts)
    avg_prec = np.mean(meta_test_prec)
    avg_rec = np.mean(meta_test_rec)
    avg_f1 = np.mean(meta_test_f1)
    print('Average prec/rec/f1: {:.4f}/{:.4f}/{:.4f}'.format(avg_prec, avg_rec, avg_f1))

from options import *
if __name__ == '__main__':
    args = get_args()
    results = run_protonet(args.data_root, n_way=args.label_subset_size, n_support=args.support_size, n_query=args.support_size, n_meta_test_way=args.label_subset_size, n_meta_test_support=args.support_size, n_meta_test_query=args.support_size, multi=args.multilabel_scheme, experiment_name=args.experiment_name, n_episodes=args.iterations, latent_dim=args.embed_dim, lr=args.lr, num_filters=args.num_conv_filters, log_frequency=args.log_frequency, patience=args.patience)

