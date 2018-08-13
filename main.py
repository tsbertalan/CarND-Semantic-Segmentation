#!/usr/bin/env python3
import warnings
import os.path
import os
import time

# Suppress unneeded tf logging.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    import tensorflow as tf

import matplotlib.pyplot as plt
import tqdm
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


def load_vgg(sess, vgg_path='vgg16'):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    with tf.variable_scope('VGG16'):
        tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
        graph = tf.get_default_graph()
    return (
        graph.get_tensor_by_name('VGG16/image_input:0'),
        graph.get_tensor_by_name('VGG16/keep_prob:0'),
        graph.get_tensor_by_name('VGG16/layer3_out:0'),
        graph.get_tensor_by_name('VGG16/layer4_out:0'),
        graph.get_tensor_by_name('VGG16/layer7_out:0'),
    )
    
    return None, None, None, None, None
# tests.test_load_vgg(load_vgg, tf)


def _get_conv2d_transpose_weights(height, width, from_tensor, out_channels):
    in_channels = from_tensor.shape[-1].value
    w = tf.Variable(tf.truncated_normal(shape=[height, width, out_channels, in_channels]))
    print()
    print('w.shape=', w.shape)
    print()
    return w


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    i1x1 = [0]
    def conv1x1(x, M=num_classes):
        with tf.variable_scope('conv1x1_%d' % i1x1[0]):
            i1x1[0] += 1
            return tf.layers.conv2d(
                x, M, 1, 1, 
                padding='SAME',
                kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
            )

    iupsample = [0]
    def upsample(x, name, M=num_classes, k=4, stride=2):
        with tf.variable_scope('upsample_%d' % iupsample[0]):
            iupsample[0] += 1
            return tf.layers.conv2d_transpose(
                x,
                M,
                k,
                strides=(stride, stride),
                padding='SAME',
                kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
            )

    # 1x1 convolution of vgg layer 7
    x = conv1x1(vgg_layer7_out)

    # upsample
    x = upsample(x, 'layer4a_in1')

    # skip connection (element-wise addition)
    x = tf.add(x, conv1x1(vgg_layer4_out))

    # upsample
    x = upsample(x, 'layer3a_in1')

    # skip connection (element-wise addition)
    x = tf.add(x, conv1x1(vgg_layer3_out))

    # upsample
    x = upsample(x, 'nn_last_layer', k=16, stride=8)

    return x

# with warnings.catch_warnings():
#     warnings.simplefilter('ignore')
#     tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
# tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    results = []

    try:
        pbar = tqdm.tqdm(total=epochs*get_batches_fn.nbatches(batch_size))
        update = lambda : pbar.update()
    except AttributeError:
        update = lambda : None

    try:
        for epoch in range(epochs):
            for image, label in get_batches_fn(batch_size):
                loss_value = sess.run(
                        [train_op, cross_entropy_loss],
                        feed_dict={
                            input_image: image, 
                            correct_label: label, 
                            keep_prob: .5, 
                            learning_rate: 1e-3,
                        }
                    )[1]
                results.append(loss_value)
                update()
    except KeyboardInterrupt:
        pass
    return results
# tests.test_train_nn(train_nn)


def graph2pdf(sess, directory, **kw):
    import tfgraphviz as tfg
    g = tfg.board(sess.graph, **kw)
    g.render(filename='graph', directory=directory)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = '/home/tsbertalan/data/'
    runs_dir = './runs'
    # tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(
            os.path.join(data_dir, 'data_road/training'), 
            image_shape,
            #maxdata=16,
        )

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)

        # Save graph picture to file.
        tag = str(time.time())
        directory = os.path.join(runs_dir, tag)
        graph2pdf(sess, directory, depth=1)

        correct_label = tf.placeholder('float32', shape=[None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder('float32', name='learning_rate')

        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        sess.run(tf.global_variables_initializer())
        
        # Train NN using the train_nn function
        train_losses = train_nn(
            sess,
            epochs=50, batch_size=8, get_batches_fn=get_batches_fn, 
            train_op=train_op, cross_entropy_loss=cross_entropy_loss, input_image=input_image,
            correct_label=correct_label, keep_prob=keep_prob, learning_rate=learning_rate,
        )

        # Save inference data using helper.save_inference_samples
        output_dir = helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, tag)

        fig, ax = plt.subplots()
        ax.plot(train_losses)
        ax.set_yscale('log')
        ax.set_ylabel('loss')
        ax.set_xlabel('batches')
        fig.savefig(output_dir + '/losshist.png')


        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
    #plt.show()
