import tensorflow as tf

# Inner Model
def conv_block(inp, cweight, bweight, bn, activation=tf.nn.relu, residual=False):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1, 2, 2, 1], [1, 1, 1, 1]

    conv_output = tf.nn.conv2d(input=inp, filters=cweight, strides=no_stride, padding='SAME') + bweight
    normed = bn(conv_output)
    normed = activation(normed)
    return normed

class VGGWrapper(tf.keras.layers.Layer):
    def __init__(self, channels, dim_hidden, dim_output, img_size):
        super(VGGWrapper, self).__init__()
        self.channels = channels
        self.dim_output = dim_output
        self.img_size = img_size
        self.model = tf.keras.applications.VGG19(include_top=False, pooling='max', input_shape=(self.img_size, self.img_size, self.channels))
        vgg_output_shape = self.model.output_shape
        self.vgg_weight_names = [layer.name for layer in self.model.trainable_variables]
        weight_initializer = tf.keras.initializers.GlorotUniform()
        self.model_weights = dict([(layer.name, layer) for layer in self.model.trainable_weights])
        self.model_weights['dense:weights'] = tf.Variable(weight_initializer(shape=[vgg_output_shape[-1], self.dim_output]), name='dense:weights')
        self.model_weights['dense:bias'] = tf.Variable(tf.zeros([self.dim_output]), name='dense:bias')

    def __call__(self, inp, weights):
        inp = tf.transpose(inp, perm=[0, 2, 3, 1])
        # populate VGG structure with these weights
        vgg_weights = [tensor for w, tensor in weights.items() if w in self.vgg_weight_names]
        temp_model = tf.keras.models.clone_model(self.model)

        temp_model.set_weights(vgg_weights)
        out = temp_model(inp)
        return tf.matmul(out, weights['dense:weights']) + weights['dense:bias']


class VanillaConvModel(tf.keras.layers.Layer):
    def __init__(self, channels, dim_hidden, dim_output, img_size):
        super(VanillaConvModel, self).__init__()
        self.channels = channels
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.img_size = img_size

        weights = {}

        dtype = tf.float32
        weight_initializer = tf.keras.initializers.GlorotUniform()
        k = 3

        weights['conv1'] = tf.Variable(weight_initializer(shape=[k, k, self.channels, self.dim_hidden]), name='conv1', dtype=dtype)
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b1')
        self.bn1 = tf.keras.layers.BatchNormalization(name='bn1')
        weights['conv2'] = tf.Variable(weight_initializer(shape=[k, k, self.dim_hidden, self.dim_hidden]), name='conv2', dtype=dtype)
        weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b2')
        self.bn2 = tf.keras.layers.BatchNormalization(name='bn2')
        weights['conv3'] = tf.Variable(weight_initializer(shape=[k, k, self.dim_hidden, self.dim_hidden]), name='conv3', dtype=dtype)
        weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b3')
        self.bn3 = tf.keras.layers.BatchNormalization(name='bn3')
        weights['conv4'] = tf.Variable(weight_initializer([k, k, self.dim_hidden, self.dim_hidden]), name='conv4', dtype=dtype)
        weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b4')
        self.bn4 = tf.keras.layers.BatchNormalization(name='bn4')
        weights['w5'] = tf.Variable(weight_initializer(shape=[self.dim_hidden, self.dim_output]), name='w5', dtype=dtype)
        weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        self.model_weights = weights

    def call(self, inp, weights):
        channels = self.channels
        inp = tf.transpose(inp, perm=[0, 2, 3, 1])
        #inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])
        hidden1 = conv_block(inp, weights['conv1'], weights['b1'], self.bn1)
        hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], self.bn2)
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], self.bn3)
        hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], self.bn4)
        hidden4 = tf.reduce_mean(input_tensor=hidden4, axis=[1, 2])
        return tf.matmul(hidden4, weights['w5']) + weights['b5']


