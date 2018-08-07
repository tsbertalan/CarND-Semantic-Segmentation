#!/usr/bin/env python3
import warnings
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path='vgg16'):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    graph = tf.get_default_graph()
    return (
        graph.get_tensor_by_name('image_input:0'),
        graph.get_tensor_by_name('keep_prob:0'),
        graph.get_tensor_by_name('layer3_out:0'),
        graph.get_tensor_by_name('layer4_out:0'),
        graph.get_tensor_by_name('layer7_out:0'),
    )
    
    return None, None, None, None, None
tests.test_load_vgg(load_vgg, tf)


def _get_conv2d_transpose_weights(height, width, from_tensor, out_channels):
    in_channels = from_tensor.shape[-1].value
    w = tf.Variable(tf.truncated_normal(shape=[height, width, out_channels, in_channels]))
    print()
    print('w.shape=', w.shape)
    print()
    return w

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  
    Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # Do one or more 1x1 convolutions for our "decision-making" section.
    conv_1x1 = tf.layers.conv2d(
        vgg_layer7_out, 10, 1, padding='SAME', 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3)
    )
    # conv_1x1 = tf.layers.conv2d(
    #     conv_1x1, 10, 1, padding='SAME', 
    #     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3)
    # )

    # Upsample to layer 3 and add.
    upsampled_3 = tf.layers.conv2d_transpose(
        conv_1x1,
        vgg_layer3_out.shape[-1].value,
        [4, 4],
        strides=[2, 2],
        padding='SAME',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
        name='upsampled_3',
    )
    skipped_3 = tf.add(upsampled_3, vgg_layer3_out)

    # Upsample to layer 4 and add.
    upsampled_4 = tf.layers.conv2d_transpose(
        skipped_3,
        vgg_layer4_out.shape[-1].value,
        [4, 4],
        strides=[2, 2],
        padding='SAME',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
        name='upsampled_4',
    )
    skipped_4 = tf.add(upsampled_4, vgg_layer4_out)

    # Upsample to layer 7 and add.
    upsampled_7 = tf.layers.conv2d_transpose(
        skipped_4,
        vgg_layer7_out.shape[-1].value,
        [16, 16],
        strides=[8, 8],
        padding='SAME',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
        name='upsampled_7',
    )
    skipped_7 = tf.add(upsampled_7, vgg_layer7_out)

    # Finally, do a 1x1 convolution to the right number of output channels.
    return tf.layers.conv2d(
        skipped_7, num_classes, 1, padding='SAME',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
    )

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    tests.test_layers(layers)


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
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


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
    for epoch in range(epochs):
        for image, label in get_batches_fn(batch_size):
            results.append(
                sess.run(
                    [train_op, cross_entropy_loss],
                    feed_dict={
                        input_image: image, 
                        correct_label: label, 
                        keep_prob: .5, 
                        learning_rate: 1e-3,
                    }
                )
            )
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = '/home/tsbertalan/data/'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)

        correct_label = tf.placeholder('float32', shape=[None, image_shape[0], image_shape[1], num_classes], name='correct_label')
        learning_rate = tf.placeholder('float32', name='learning_rate')

        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        sess.run(tf.global_variables_initializer())
        
        # Train NN using the train_nn function
        train_results = train_nn(
            sess,
            epochs=10, batch_size=1, get_batches_fn=get_batches_fn, 
            train_op=train_op, cross_entropy_loss=cross_entropy_loss, input_image=input_image,
            correct_label=correct_label, keep_prob=keep_prob, learning_rate=learning_rate,
        )

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
