import tensorflow as tf
from util import *
import random

CROP_SIZE = 224

img_width = 256
img_heigh = 256

# --------------------------------- LALER FUNCTION ----------------------------------------- #
def Conv2d(batch_input, n_fiter, filter_size, strides, act=None, padding='SAME', name='conv'):
    with tf.variable_scope(name):
        in_channels = batch_input.get_shape()[3]
        filters = tf.get_variable('filter', [filter_size, filter_size, in_channels, n_fiter], dtype=tf.float32,
                                  initializer=tf.random_normal_initializer(0, 0.02))
        conv = tf.nn.conv2d(batch_input, filters, [1, strides, strides, 1], padding=padding)
        if act is not None:
            conv = act(conv)
        return conv


def Deconv(batch_input, n_fiter, filter_size, strides, act=None, padding='SAME', name='deconv'):
    with tf.variable_scope(name):
        x_shape = tf.shape(batch_input)
        output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, n_fiter])
        in_channels = batch_input.get_shape()[-1]
        filters = tf.get_variable('filter', [filter_size, filter_size, n_fiter, in_channels], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))
        conv = tf.nn.conv2d_transpose(batch_input, filters, output_shape, [1, strides, strides, 1], padding=padding)
        conv = tf.reshape(conv, output_shape)
        if act is not None:
            conv = act(conv)
        return conv


def LeakyReLu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def Batchnorm(input, act, is_train, name):
    with tf.variable_scope(name):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)
        variance_epsilon = 1e-5
        normalized = tf.contrib.layers.batch_norm(input, center=True, scale=True, epsilon=variance_epsilon,
                                                  activation_fn=act, is_training=is_train, reuse=None)
        '''
        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        '''
        return normalized


def Elementwise(n1, n2, act, name):
    with tf.variable_scope(name):
        return act(n1, n2)


# --------------------------------- MODEL DEFINITION --------------------------------- #

def input_producer(data_list, channels, batch_size, need_shuffle):
    if len(data_list) == 0:
        raise Exception("empty data list!")

    #data_list = open(data_list_file, 'rt').read().splitlines()

    def read_data(data_queue):
        # note : read one training data : pixel range : [0, 255]
        in_img = tf.image.decode_image(tf.read_file(data_queue[0]), channels=channels)
        gt_img = tf.image.decode_image(tf.read_file(data_queue[1]), channels=channels)

        def preprocessing(input):
            proc = tf.cast(input, tf.float32)
            proc.set_shape([img_width, img_heigh, channels])
            # normalization
            proc = proc / 127.5 - 1
            return proc

        # output pixel's range : [-1, 1]
        in_imgproc = preprocessing(in_img)
        gt_imgproc = preprocessing(gt_img)
        return in_imgproc, gt_imgproc

    with tf.variable_scope('input'):
        # Get full list of image and labels
        imglist = [s.split(' ')[0] for s in data_list]
        lablist = [s.split(' ')[-1] for s in data_list]
        srcfilelist = tf.convert_to_tensor(imglist, dtype=tf.string)
        dstfilelist = tf.convert_to_tensor(lablist, dtype=tf.string)

        # Put images and label into a queue
        data_queue = tf.train.slice_input_producer([srcfilelist, dstfilelist], capacity=64, shuffle=need_shuffle)

        # Read one data from queue
        input, target = read_data(data_queue)
        '''
        # testing
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            sdfa1, sdfa2 = sess.run([input, target])
            print sdfa1.shape, sdfa2.shape
        '''

        # Construct a batch of training data
        in_batch, gt_batch = tf.train.batch([input, target], batch_size, num_threads=1, capacity=64)
    return in_batch, gt_batch, len(data_list)


def encode(inputs, out_channels, is_train=False, reuse=False):
    with tf.variable_scope("encode", reuse=reuse):
        # in
        n = Conv2d(inputs, 64, filter_size=3, strides=1, padding='SAME', name='in/k3n64s1')
        #n = Batchnorm(n, act=tf.nn.relu, is_train=is_train, name='in/BN')

        # start residual blocks
        for i in range(2):
            nn = Conv2d(n, 64, filter_size=3, strides=1, act=tf.nn.relu, padding='SAME', name='sen64s1/c1/%s' % i)
            #nn = Batchnorm(nn, act=tf.nn.relu, is_train=is_train, name='sen64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, filter_size=3, strides=1, padding='SAME', name='sen64s1/c2/%s' % i)
            #nn = Batchnorm(nn, act=None, is_train=is_train, name='sen64s1/b2/%s' % i)
            nn = Elementwise(n, nn, tf.add, 'seb_residual_add/%s' % i)
            n = nn

        n1 = n
        # down size
        n = Conv2d(n1, 128, filter_size=3, strides=2, padding='SAME', name='down1-1/k3n128s2')
        n2 = Conv2d(n, 128, filter_size=3, strides=1, act=tf.nn.relu, padding='SAME', name='down1-2/k3n128s1')
        #n2 = Batchnorm(n, act=tf.nn.relu, is_train=is_train, name='down1/BN')
        n = Conv2d(n2, 256, filter_size=3, strides=2, padding='SAME', name='down2-1/k3n256s2')
        n = Conv2d(n, 256, filter_size=3, strides=1, act=tf.nn.relu, padding='SAME', name='down2-2/k3n256s1')
        #n = Batchnorm(n, act=tf.nn.relu, is_train=is_train, name='down2/BN')

        # residual blocks
        for i in range(4):
            nn = Conv2d(n, 256, filter_size=3, strides=1, act=tf.nn.relu, padding='SAME', name='en64s1/c1/%s' % i)
            #nn = Batchnorm(nn, act=tf.nn.relu, is_train=is_train, name='en64s1/b1/%s' % i)
            nn = Conv2d(nn, 256, filter_size=3, strides=1, padding='SAME', name='en64s1/c2/%s' % i)
            #nn = Batchnorm(nn, act=None, is_train=is_train, name='en64s1/b2/%s' % i)
            nn = Elementwise(n, nn, tf.add, 'eb_residual_add/%s' % i)
            n = nn

        # up size
        n = tf.image.resize_images(n, [tf.shape(n)[1] * 2, tf.shape(n)[2] * 2], method=1)
        n = Conv2d(n, 128, filter_size=3, strides=1, padding='SAME', name='up1-1/k3n128s1')
        n = Conv2d(n, 128, filter_size=3, strides=1, act=tf.nn.relu, padding='SAME', name='up1-2/k3n128s1')
        #n = Batchnorm(n, act=tf.nn.relu, is_train=is_train, name='up1/BN')
        n = Elementwise(n, n2, tf.add, 'skipping1')

        n = tf.image.resize_images(n, [tf.shape(n)[1] * 2, tf.shape(n)[2] * 2], method=1)
        n = Conv2d(n, 64, filter_size=3, strides=1, padding='SAME', name='up2-1/k3n64s1')
        n = Conv2d(n, 64, filter_size=3, strides=1, act=tf.nn.relu, padding='SAME', name='up2-2/k3n64s1')
        #n = Batchnorm(n, act=tf.nn.relu, is_train=is_train, name='up2/BN')
        n = Elementwise(n, n1, tf.add, 'skipping2')

        # end residual blocks
        for i in range(2):
            nn = Conv2d(n, 64, filter_size=3, strides=1, act=tf.nn.relu, padding='SAME', name='een64s1/c1/%s' % i)
            #nn = Batchnorm(nn, act=tf.nn.relu, is_train=is_train, name='een64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, filter_size=3, strides=1, padding='SAME', name='een64s1/c2/%s' % i)
            #nn = Batchnorm(nn, act=None, is_train=is_train, name='een64s1/b2/%s' % i)
            nn = Elementwise(n, nn, tf.add, 'eeb_residual_add/%s' % i)
            n = nn

        latent_maps = Conv2d(n, out_channels, filter_size=3, strides=1, act=tf.nn.tanh, padding='SAME', name='latent')

        return latent_maps


def decode(latents, out_channels, is_train=False, reuse=False):
    with tf.variable_scope("decode", reuse=reuse):
        # Decoder
        n = Conv2d(latents, 64, filter_size=3, strides=1, padding='SAME', name='in/k3n64s1')
        for i in range(8):
            nn = Conv2d(n, 64, filter_size=3, strides=1, act=tf.nn.relu, padding='SAME', name='dn64s1/c1/%s' % i)
            #nn = Batchnorm(nn, act=tf.nn.relu, is_train=is_train, name='dn64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, filter_size=3, strides=1, padding='SAME', name='dn64s1/c2/%s' % i)
            #nn = Batchnorm(nn, act=None, is_train=is_train, name='dn64s1/b2/%s' % i)
            nn = Elementwise(n, nn, tf.add, 'db_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, 256, filter_size=3, strides=1, act=None, padding='SAME', name='n256s1/2')
        n = Conv2d(n, out_channels, filter_size=1, strides=1, padding='SAME', name='out')
        output_map = tf.nn.tanh(n)

        return output_map


class VGG19:
    def __init__(self, vgg19_npy_path=None):
        if vgg19_npy_path is None:
            print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
            exit()
        self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        print("vgg19 npy file loaded!")

    def build(self, input, is_rgb):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """
        if is_rgb:
            rgb = input
        else:
            rgb = tf.concat([input, input, input], -1)
            shape = rgb.get_shape().as_list()
            shape[-1] = 3
            rgb.set_shape(shape)

        VGG_MEAN = [103.939, 116.779, 123.68]
        with tf.name_scope("VGG19"):
            rgb_scaled = rgb * 255.0
            # Convert RGB to BGR
            red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
            assert red.get_shape().as_list()[1:] == [224, 224, 1]
            assert green.get_shape().as_list()[1:] == [224, 224, 1]
            assert blue.get_shape().as_list()[1:] == [224, 224, 1]
            bgr = tf.concat(axis=3, values=[
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
            assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
            """ conv1 """
            nn = self.conv_layer(bgr, "conv1_1")
            nn = self.conv_layer(nn, "conv1_2")
            nn = self.max_pool(nn, 'pool1')
            """ conv2 """
            nn = self.conv_layer(nn, "conv2_1")
            nn = self.conv_layer(nn, "conv2_2")
            nn = self.max_pool(nn, 'pool2')
            """ conv3 """
            nn = self.conv_layer(nn, "conv3_1")
            nn = self.conv_layer(nn, "conv3_2")
            nn = self.conv_layer(nn, "conv3_3")
            nn = self.conv_layer(nn, "conv3_4")
            nn = self.max_pool(nn, 'pool3')
            """ conv4 """
            nn = self.conv_layer(nn, "conv4_1")
            # conv4_1
            feature_map = nn
            return feature_map
            nn = self.conv_layer(nn, "conv4_2")
            nn = self.conv_layer(nn, "conv4_3")
            nn = self.conv_layer(nn, "conv4_4")
            nn = self.max_pool(nn, 'pool4')


            """ conv5 """
            nn = self.conv_layer(nn, "conv5_1")
            nn = self.conv_layer(nn, "conv5_2")
            nn = self.conv_layer(nn, "conv5_3")
            nn = self.conv_layer(nn, "conv5_4")
            nn = self.max_pool(nn, 'pool5')

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")



