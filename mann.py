import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers

import os

import load_data_tf as load_data
from options import get_args
from utils import convert_to_powerset, generate_experiment_name, precision, recall, fscore, convert_to_bin_rel
import time
from tensorboardX import SummaryWriter

#tf.keras.backend.set_floatx('float64')
np.set_printoptions(precision=4)

class SNAILConvBlock(tf.keras.Model):
    def __init__(self, filters=64, batch_norm=True, pool='max', pool_size=(2, 2)):
        super(SNAILConvBlock, self).__init__()
        self.batch_norm = batch_norm
        self.pool_type = pool
        self.conv = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        if pool in ['max', 'maximum']:
            self.pool = tf.keras.layers.MaxPooling2D(pool_size=pool_size)
        elif pool in ['average', 'avg']:
            self.pool =  tf.keras.layers.AveragePooling2D(pool_size=pool_size)
        else:
            self.pool = None

    def call(self, inp):
        out = self.conv(inp)
        if self.batch_norm: out = self.bn(out)
        if self.pool: out = self.pool(out)

        return out

class MANN(tf.keras.Model):

    def __init__(self, num_classes, support_size, query_size, num_blocks=1, embed_size=64, memory_size=512, multi = 'powerset'):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.support_size = support_size
        self.query_size = query_size
        #self.samples_per_class = samples_per_class
        #self.conv1 = tf.keras.layers.Conv2D(1, 3, activation='relu', padding="same")
        #self.conv2 = tf.keras.layers.Conv2D(1, 3, activation='relu', padding="same")
        self.blocks = [SNAILConvBlock() for _ in range(num_blocks)]
        self.dense = tf.keras.layers.Dense(embed_size)
        self.multi = multi

        if self.multi == 'binary':
            self.lower_lstm = []
            self.upper_lstm = []
            for _ in range(num_classes):
                self.lower_lstm.append(tf.keras.layers.LSTM(memory_size, return_sequences=True))
                self.upper_lstm.append(tf.keras.layers.LSTM(2, return_sequences=True))
        else:
            self.lower_lstm = tf.keras.layers.LSTM(memory_size, return_sequences=True)
            self.upper_lstm = tf.keras.layers.LSTM(num_classes, return_sequences=True)

    def call(self, input_images, input_labels):
        b, s, h, w, c = input_images.shape
        conv_out = tf.reshape(input_images, (-1, w, h, c))
        for block in self.blocks:
            conv_out = block(conv_out)
        #hidden1 = self.conv1(conv_input)
        #conv_out = self.conv2(hidden1) # this is (b, s, f, w, h)
        _, f_, h_, w_ = conv_out.shape
        img_reshaped = tf.reshape(conv_out, (-1, f_, h_, w_)) # (b, s, f * w * h)

        img_reshaped = tf.reshape(img_reshaped, (b, s, -1))
        img_reshaped = self.dense(img_reshaped)

        lbl_reshaped = tf.concat((input_labels[:, :self.support_size, :], tf.zeros_like(input_labels[:, self.support_size:, :])), axis=1)
        #lbl_reshaped = tf.reshape(lbl_reshaped, (-1, n*k, n)) 

        #SPLIT HERE, CONCAT BINARY LAYER


        if self.multi == 'binary':
            out = tf.repeat(tf.expand_dims(img_reshaped, axis=-2), self.num_classes, axis=-2)
            out = tf.concat((out, lbl_reshaped), axis=-1)  # (b, s, n, e + 2)
            class_splits = tf.split(out, self.num_classes, axis=-2)
            out = [self.lower_lstm[i](tf.squeeze(class_splits[i], axis=-2)) for i in range(self.num_classes)]
            out = [self.upper_lstm[i](out[i]) for i in range(self.num_classes)]
            #out = [lower_layer(out) for lower_layer in self.lower_lstm]
            #out = [upper_layer(out) for upper_layer in self.upper_lstm]
            out = tf.stack(out, axis=-2)
        else:
            x = tf.concat((img_reshaped, lbl_reshaped), axis=-1) # (b, s, n + e)
            out = self.lower_lstm(x)
            out = self.upper_lstm(out)
            #out = tf.reshape(out, shape)

        #############################
        return out

    def loss_function(self, preds, labels):

        #############################
        #### YOUR CODE GOES HERE #### 
        #tf.print(preds[0], labels[0], summarize=-1)
        y_pred = preds[:, self.support_size:, ...]
        y_true = labels[:, self.support_size:, ...]
        # shape is [b, q, powerset_size] or [b, q, n_classes, 2]
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        if self.multi == 'powerset':
            loss = cce(y_true, y_pred)
        else:
            class_losses = [cce(class_true, class_pred) for class_true, class_pred in zip(tf.split(y_true, self.num_classes, axis=-2), tf.split(y_pred, self.num_classes, axis=-2))]
            loss = tf.reduce_mean(class_losses)
        return loss
        #############################


@tf.function
def train_step(images, labels, model, optim, eval=False):
    with tf.GradientTape() as tape:
        predictions = model(images, labels)
        loss = model.loss_function(predictions, labels)
    if not eval:
        gradients = tape.gradient(loss, model.trainable_variables)
        optim.apply_gradients(zip(gradients, model.trainable_variables))
    return predictions, loss


def main(data_root='../cs330-storage/', num_classes=3, support_size=16, query_size=4, meta_batch_size=8, random_seed=42, iterations=1000, experiment_name=None, lr=1e-3, lr_schedule=False, sampling_mode='greedy', multi = 'powerset'):
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    filter_files = [os.path.join(data_root, 'patches_with_cloud_and_shadow.csv'), os.path.join(data_root, 'patches_with_seasonal_snow.csv')]  # replace with your path
    data_dir = os.path.join(data_root, "SmallEarthNet")
    meta_dataset = load_data.MetaBigEarthNetTaskDataset(data_dir=data_dir, filter_files=filter_files, support_size=support_size + query_size, label_subset_size=num_classes, split_save_path="smallearthnet.pkl", split_file="smallearthnet.pkl", data_format='channels_last')

    experiment_fullname = generate_experiment_name(experiment_name)
    log_dir = '../tensorboard_logs/' + experiment_fullname
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)


    if multi == 'powerset': num_classes = (1 << num_classes) - 1
    o = MANN(num_classes, support_size, query_size, multi = multi)

    lr_config = lr
    if lr_schedule:
        lr_config = tf.keras.optimizers.schedules.ExponentialDecay(
          initial_learning_rate=lr,
          decay_steps=1000,
          decay_rate=0.85,
          staircase=True)
    #optim = tf.keras.optimizers.Adam(learning_rate=0.00001)
    optim = tf.keras.optimizers.SGD(learning_rate=lr_config)
    test_accuracy = []
    for step in range(iterations):
        start = time.time()
        X, y, y_debug = meta_dataset.sample_batch(batch_size=meta_batch_size, split='train', mode=sampling_mode)
        y = convert_to_powerset(y) if multi == 'powerset' else convert_to_bin_rel(y)
        #print(y.shape)
        _, ls = train_step(X, y, o, optim)

        if (step + 1) % 1 == 0:
            X, y, y_debug = meta_dataset.sample_batch(batch_size=meta_batch_size, split='validation', mode=sampling_mode)
            raw_y = convert_to_powerset(y) if multi == 'powerset' else convert_to_bin_rel(y)
            raw_pred, tls = train_step(X, raw_y, o, optim, eval=True)

            pred_tr = tf.math.argmax(raw_pred[:, :support_size, :], axis=-1)
            y_tr = tf.math.argmax(raw_y[:, :support_size, :], axis=-1)
            pred_ts = tf.math.argmax(raw_pred[:, support_size:, :], axis=-1)
            y_ts = tf.math.argmax(raw_y[:, support_size:, :], axis=-1)

            shape = (-1,) if multi == 'powerset' else (-1, num_classes)
            pred_tr = tf.reshape(pred_tr, shape)
            y_tr = tf.reshape(y_tr, shape)
            pred_ts = tf.reshape(pred_ts, shape)
            y_ts = tf.reshape(y_ts, shape)

            train_acc = tf.reduce_mean(tf.cast(tf.math.equal(pred_tr, y_tr), tf.float32)).numpy()
            test_acc = tf.reduce_mean(tf.cast(tf.math.equal(pred_ts, y_ts), tf.float32)).numpy()
            test_accuracy.append(test_acc)
            prec_tr = precision(y_tr, pred_tr, multi)
            rec_tr = recall(y_tr, pred_tr, multi)
            f1_tr = fscore(y_tr, pred_tr, multi)
            prec_ts = precision(y_ts, pred_ts, multi)
            rec_ts = recall(y_ts, pred_ts, multi)
            f1_ts = fscore(y_ts, pred_ts, multi)

            # debug only
            #full_preds = tf.math.argmax(raw_pred, axis=-1)
            #full_y = tf.math.argmax(raw_y, axis=-1)
            print("Iteration {}/{} -- Train Loss: {:.4f}".format(step + 1, iterations, ls.numpy()), "Test Loss: {:.4f}".format(tls.numpy()), "Test Prec/Rec/F1: {:.4f}/{:.4f}/{:.4f}".format(prec_ts, rec_ts, f1_ts), "Time: {:.4f}s".format(time.time() - start))
            writer.add_scalar("Train loss", ls.numpy(), step)
            writer.add_scalar("Test loss", tls.numpy(), step)
            writer.add_scalar("Test accuracy", test_acc, step)
            writer.add_scalar("Test precision", prec_tr.numpy(), step)
            writer.add_scalar("Test recall", rec_tr.numpy(), step)
            writer.add_scalar("Test F1", f1_tr.numpy(), step)
            writer.add_scalar("Test accuracy", train_acc, step)
            writer.add_scalar("Test precision", prec_ts.numpy(), step)
            writer.add_scalar("Test recall", rec_ts.numpy(), step)
            writer.add_scalar("Test F1", f1_ts.numpy(), step)
    return test_accuracy

if __name__ == '__main__':
    args = get_args()
    main(data_root=args.data_root, iterations=args.iterations, support_size=args.support_size, num_classes=args.label_subset_size, experiment_name=args.experiment_name, meta_batch_size=args.bs, sampling_mode=args.sampling_mode, multi = args.multilabel_scheme)


