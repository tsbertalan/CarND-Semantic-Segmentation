import os, warnings, time
# Suppress unneeded tf logging.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import tensorflow as tf
    import project_tests as tests
    import helper

from main import load_vgg, layers, optimize, train_nn

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = '/home/tsbertalan/data/'
    runs_dir = './runs'
    # tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    with tf.Session() as sess:

        tag = str(time.time())
        directory = os.path.join(runs_dir, tag)

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(
            os.path.join(data_dir, 'data_road/training'), 
            image_shape,
            num_classes=num_classes,
            #maxdata=16,
            sample_aug_folder=os.path.join(directory, 'augmented_samples')
        )

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)

        tests.test_load_vgg(load_vgg, tf)
        tests.test_layers(layers)
        tests.test_optimize(optimize)
        tests.test_train_nn(train_nn)

if __name__ == '__main__':
    run()
