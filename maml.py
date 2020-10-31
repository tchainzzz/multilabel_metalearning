###################
# TODO: -create separate utils and models files
#	   -command line args
#	   		-verbose
#			-num iteatrations
#			-learning rate
#			-dynamic learning rate
#		-pretrained model
#		-implement binary relevance
#
##################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


import random
import sys
import csv
import pickle
import time
#import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from functools import partial

import load_data_tf as load_data
from models import VanillaConvModel
from utils import *
from options import get_args

import sys
import logging


seed = 123
IMG_SIZE = 120

class MAML(tf.keras.Model):
    def __init__(self, dim_input=1, dim_output=1, num_inner_updates=1, inner_update_lr=0.4, num_filters=32, learn_inner_update_lr=False):
        super(MAML, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.inner_update_lr = inner_update_lr
        self.loss_func = partial(cross_entropy_loss)
        self.dim_hidden = num_filters
        self.channels = 3
        self.img_size = int(np.sqrt(self.dim_input / self.channels))

        # outputs_ts[i] and losses_ts_post[i] are the output and loss after i+1
        # inner gradient updates
        losses_tr_pre, outputs_tr, losses_ts_post, outputs_ts = [], [], [], []
        accuracies_tr_pre, accuracies_ts = [], []

        # for each loop in the inner training loop
        outputs_ts = [[]] * num_inner_updates
        losses_ts_post = [[]] * num_inner_updates
        accuracies_ts = [[]] * num_inner_updates

        # Define the weights - these should NOT be directly modified by the
        # inner training loop
        tf.random.set_seed(seed)
        self.conv_layers = VanillaConvModel(self.channels, self.dim_hidden, self.dim_output, self.img_size)

        self.learn_inner_update_lr = learn_inner_update_lr
        if self.learn_inner_update_lr:
            self.inner_update_lr_dict = {}
            for key in self.conv_layers.conv_weights.keys():
                self.inner_update_lr_dict[key] = [
                    tf.Variable(self.inner_update_lr, name='inner_update_lr_%s_%d' %
                        (key, j)) for j in range(num_inner_updates)]

    @tf.function
    def call(self, inp, meta_batch_size=25, num_inner_updates=1):
        def task_inner_loop(inp, reuse=True,
                            meta_batch_size=25, num_inner_updates=1):
            start = time.time()
            # the inner and outer loop data
            input_tr, input_ts, label_tr, label_ts = inp

            # weights corresponds to the initial weights in MAML 
            weights = self.conv_layers.conv_weights

            # the predicted outputs, loss values, and accuracy for the pre-update model (with the initial weights), evaluated on the inner loop training data
            task_output_tr_pre, task_loss_tr_pre, task_accuracy_tr_pre = None, None, None

            # lists to keep track of outputs, losses, and accuracies of test data for each inner_update
            # where task_outputs_ts[i], task_losses_ts[i], task_accuracies_ts[i] are the output, loss, and accuracy after i+1 inner gradient updates
            task_outputs_ts, task_losses_ts, task_accuracies_ts = [], [], []

            task_output_tr_pre = self.conv_layers(input_tr, weights)
            task_loss_tr_pre = self.loss_func(task_output_tr_pre, label_tr)
            for i in range(num_inner_updates):
                with tf.GradientTape(persistent=False) as tape: # keep track of high-order derivs on train data only, and use those to update
                    for key in weights: tape.watch(weights[key])
                    output_tr = self.conv_layers(input_tr, weights)
                    loss_tr = self.loss_func(output_tr, label_tr)
                grads = dict(zip(weights.keys(), tape.gradient(loss_tr, list(weights.values())))) # might need to make a TF op 
                if self.learn_inner_update_lr:
                    weights = dict(zip(weights.keys(), [weights[key] - self.inner_update_lr_dict[key] * grads[key] for key in weights])) # might need to make a TF op? probably OK
                else:
                    weights = dict(zip(weights.keys(), [weights[key] - self.inner_update_lr * grads[key] for key in weights])) # might need to make a TF op? probably OK

                # post-update metrics on test
                output_ts = self.conv_layers(input_ts, weights)
                task_outputs_ts.append(output_ts) # TODO: Make TF op
                task_losses_ts.append(self.loss_func(output_ts, label_ts)) # TODO: Make TF op

            #############################

            # Compute accuracies from output predictions
            task_accuracy_tr_pre = accuracy(tf.argmax(input=label_tr, axis=1), tf.argmax(input=tf.nn.softmax(task_output_tr_pre), axis=1))

            for j in range(num_inner_updates):
                task_accuracies_ts.append(accuracy(tf.argmax(input=label_ts, axis=1), tf.argmax(input=tf.nn.softmax(task_outputs_ts[j]), axis=1)))

            task_output = [task_output_tr_pre, task_outputs_ts, task_loss_tr_pre, task_losses_ts, task_accuracy_tr_pre, task_accuracies_ts]
            #tf.print("Iteration took {}s".format(time.time() - start))
            return task_output

        input_tr, input_ts, label_tr, label_ts = inp
        # to initialize the batch norm vars, might want to combine this, and
        # not run idx 0 twice.
        unused = task_inner_loop((input_tr[0], input_ts[0], label_tr[0], label_ts[0]), False, meta_batch_size, num_inner_updates)
        out_dtype = [tf.float32, [tf.float32] * num_inner_updates,
                     tf.float32, [tf.float32] * num_inner_updates]
        out_dtype.extend([tf.float32, [tf.float32] * num_inner_updates])
        task_inner_loop_partial = partial(task_inner_loop, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)
        result = tf.map_fn(task_inner_loop_partial,
                           elems=(input_tr, input_ts, label_tr, label_ts),
                           fn_output_signature=out_dtype,
                           parallel_iterations=meta_batch_size)
        return result


"""Model training code"""
"""
Usage Instructions:
	5-way, 1-shot omniglot:
	python main.py --meta_train_iterations=15000 --meta_batch_size=25 --k_shot=1 --inner_update_lr=0.4 --num_inner_updates=1 --logdir=logs/omniglot5way/
	20-way, 1-shot omniglot:
	python main.py --meta_train_iterations=15000 --meta_batch_size=16 --k_shot=1 --n_way=20 --inner_update_lr=0.1 --num_inner_updates=5 --logdir=logs/omniglot20way/
	To run evaluation, use the '--meta_train=False' flag and the '--meta_test_set=True' flag to use the meta-test set.
	"""


def outer_train_step(inp, model, optim, meta_batch_size=25, num_inner_updates=1):
    with tf.GradientTape(persistent=False) as outer_tape:
        result = model(inp, meta_batch_size=meta_batch_size,
                       num_inner_updates=num_inner_updates)

        outputs_tr, outputs_ts, losses_tr_pre, losses_ts, accuracies_tr_pre, accuracies_ts = result

        total_losses_ts = [tf.reduce_mean(loss_ts) for loss_ts in losses_ts]

    gradients = outer_tape.gradient(
        total_losses_ts[-1], model.trainable_variables)
    optim.apply_gradients(zip(gradients, model.trainable_variables))

    total_loss_tr_pre = tf.reduce_mean(losses_tr_pre)
    total_accuracy_tr_pre = tf.reduce_mean(accuracies_tr_pre)
    total_accuracies_ts = [tf.reduce_mean(
        accuracy_ts) for accuracy_ts in accuracies_ts]

    return outputs_tr, outputs_ts, total_loss_tr_pre, total_losses_ts, total_accuracy_tr_pre, total_accuracies_ts


def outer_eval_step(inp, model, meta_batch_size=25, num_inner_updates=1):
    result = model(inp, meta_batch_size=meta_batch_size,
                   num_inner_updates=num_inner_updates)

    outputs_tr, outputs_ts, losses_tr_pre, losses_ts, accuracies_tr_pre, accuracies_ts = result

    total_loss_tr_pre = tf.reduce_mean(losses_tr_pre)
    total_losses_ts = [tf.reduce_mean(loss_ts) for loss_ts in losses_ts]

    total_accuracy_tr_pre = tf.reduce_mean(accuracies_tr_pre)
    total_accuracies_ts = [tf.reduce_mean(
        accuracy_ts) for accuracy_ts in accuracies_ts]

    return outputs_tr, outputs_ts, total_loss_tr_pre, total_losses_ts, total_accuracy_tr_pre, total_accuracies_ts


def meta_train_fn(model, exp_string, meta_dataset, support_size=8, num_classes=7, meta_train_iterations=15000, meta_batch_size=16, log=True, logdir='/tmp/data', num_inner_updates=1, meta_lr=0.001, log_frequency=5, test_log_frequency=25):


    pre_accuracies, post_accuracies = [], []
    pre_loss, post_loss = [], []
    #num_classes = data_generator.num_classes

    optimizer = tf.keras.optimizers.Adam(learning_rate=meta_lr)

    plot_accuracies = []
    for itr in range(meta_train_iterations):
        #############################
        #### YOUR CODE GOES HERE ####

        # sample a batch of training data and partition into
        # the support/training set (input_tr, label_tr) and the query/test set (input_ts, label_ts)
        # NOTE: The code assumes that the support and query sets have the same
        # number of examples.

        X, y, y_debug = meta_dataset.sample_batch(
            batch_size=meta_batch_size, split='train')
        X = tf.reshape(X, [meta_batch_size, support_size, -1])
        input_tr, input_ts = tf.split(X, 2, axis=1)
        single_labels = (np.packbits(y.astype(int), 2, 'little') - 1).reshape((len(y), -1))
        one_hot = np.eye(num_classes)[single_labels]
        label_tr, label_ts = tf.split(one_hot, 2, axis=1)

        #############################

        inp = (input_tr, input_ts, label_tr, label_ts)
        start = time.time()
        result = outer_train_step(inp, model, optimizer, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

        pre_accuracies.append(result[-2])
        post_accuracies.append(result[-1][-1])
        pre_loss.append(result[-4])
        post_loss.append(result[-3][-1])

        if (itr != 0) and itr % log_frequency == 0:
            print_str = 'Iteration %d: pre-inner-loop train loss/accuracy: %.5f/%.5f, post-inner-loop validation loss/accuracy: %.5f/%.5f, time elapsed: %.4fs' % (itr, np.mean(pre_loss), np.mean(pre_accuracies), np.mean(post_loss), np.mean(post_accuracies), time.time() - start)
            print(print_str)
            pre_accuracies, post_accuracies = [], []
            pre_loss, post_loss = [], []

        if (itr != 0) and itr % test_log_frequency == 0:

            """
            sample a batch of validation data and partition it into
            the support/training set (input_tr, label_tr) and the query/test set (input_ts, label_ts)
            NOTE: The code assumes that the support and query sets have the
            same number of examples.
            """

            X, y, y_debug = meta_dataset.sample_batch(batch_size=meta_batch_size, split='val')
            X = tf.reshape(X, [meta_batch_size, support_size, -1])
            input_tr, input_ts = tf.split(X, 2, axis=1)
            single_labels = (np.packbits(y.astype(int), 2,'little') - 1).reshape((len(y), -1))
            one_hot = np.eye(num_classes)[single_labels]
            label_tr, label_ts = tf.split(one_hot, 2, axis=1)

            inp = (input_tr, input_ts, label_tr, label_ts)
            result = outer_eval_step(inp, model, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

            print('Meta-validation pre-inner-loop train accuracy: %.5f, meta-validation post-inner-loop test accuracy: %.5f' % (result[-2], result[-1][-1]))

            plot_accuracies.append(result[-1][-1])

    #plt.plot(np.arange(50, meta_train_iterations, 50), plot_accuracies)
    #plt.ylabel('Validation Accuracy')
    #plt.title('Question 1.4')
    #plt.show()

    model_file = logdir + '/' + exp_string + '/model' + str(itr)
    print("Saving to ", model_file)
    model.save_weights(model_file)


# calculated for omniglot
NUM_META_TEST_POINTS = 600


def meta_test_fn(model, data_generator, support_size=8, num_classes=7, meta_batch_size=25, num_inner_updates=1):

    #num_classes = data_generator.num_classes

    np.random.seed(1)
    random.seed(1)

    meta_test_accuracies = []

    for _ in range(NUM_META_TEST_POINTS):


        # sample a batch of test data and partition it into
        # the support/training set (input_tr, label_tr) and the query/test set (input_ts, label_ts)
        # NOTE: The code assumes that the support and query sets have the same
        # number of examples.

        X, y, y_debug = meta_dataset.sample_batch(batch_size=meta_batch_size, split='test')
        X = tf.reshape(X, [meta_batch_size, support_size, -1])
        input_tr, input_ts = tf.split(X, 2, axis=1)
        single_labels = (np.packbits(y.astype(int), 2, 'little') - 1).reshape((len(y), -1))
        one_hot = np.eye(num_classes)[single_labels]
        label_tr, label_ts = tf.split(one_hot, 2, axis=1)

        #############################
        inp = (input_tr, input_ts, label_tr, label_ts)
        result = outer_eval_step(inp, model, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

        meta_test_accuracies.append(result[-1][-1])

    meta_test_accuracies = np.array(meta_test_accuracies)
    means = np.mean(meta_test_accuracies)
    stds = np.std(meta_test_accuracies)
    ci95 = 1.96 * stds / np.sqrt(NUM_META_TEST_POINTS)

    print('Mean meta-test accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))


def run_maml(support_size=8, meta_batch_size=4, meta_lr=0.001, inner_update_lr=0.4, num_filters=32, num_inner_updates=1, learn_inner_update_lr=False, resume=False, resume_itr=0, log=True, logdir='./checkpoints', data_path="../SmallEarthNet", meta_train=True, meta_train_iterations=15000, meta_train_inner_update_lr=-1, label_subset_size=3, log_frequency=5, test_log_frequency=25):

    # call data_generator and get data with k_shot*2 samples per class
    #  TODO: if args.multilabel_scheme == 'powerset'
    num_classes = 2**label_subset_size - 1

    filter_files = ['../patches_with_cloud_and_shadow.csv', '../patches_with_seasonal_snow.csv']  # replace with your path



    logging.info("Loading data...")
    meta_dataset = load_data.MetaBigEarthNetTaskDataset(data_dir=data_path, filter_files=filter_files, support_size=2 * support_size, label_subset_size=label_subset_size, split_save_path="smallearthnet.pkl", split_file="smallearthnet.pkl")

    # set up MAML model
    dim_output = num_classes
    dim_input = (IMG_SIZE**2) * 3

    logging.info("Building MAML model...")
    model = MAML(dim_input, dim_output, num_inner_updates=num_inner_updates, inner_update_lr=inner_update_lr, num_filters=num_filters, learn_inner_update_lr=learn_inner_update_lr)

    if meta_train_inner_update_lr == -1:
        meta_train_inner_update_lr = inner_update_lr

    exp_string = 'supsize_' + str(support_size) + '.mbs_' + str(meta_batch_size) + '.inner_numstep_' + str(
        num_inner_updates) + '.inner_updatelr_' + str(meta_train_inner_update_lr) + '.learn_inner_update_lr_' + str(learn_inner_update_lr)

    logging.info("Train/test starts.")
    if meta_train:
        meta_train_fn(model, exp_string, meta_dataset, support_size, num_classes, meta_train_iterations, meta_batch_size, log, logdir, num_inner_updates, meta_lr, log_frequency=log_frequency, test_log_frequency=test_log_frequency)
    else:
        meta_batch_size = 1

        model_file = tf.train.latest_checkpoint(logdir + '/' + exp_string)
        print("Restoring model weights from ", model_file)
        model.load_weights(model_file)

        meta_test_fn(model, meta_dataset, support_size, num_classes, meta_batch_size, num_inner_updates)


def main(args):
    run_maml(support_size=args.support_size, inner_update_lr=args.inner_update_lr, num_inner_updates=args.num_inner_updates, meta_train_iterations=args.iterations, learn_inner_update_lr=args.learn_inner_lr, meta_train=True, label_subset_size=args.label_subset_size, log_frequency=args.log_frequency, test_log_frequency=args.test_log_frequency)

if __name__ == '__main__':
    args = get_args()
    main(args)
