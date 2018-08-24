import re
import random
import numpy as np
import os, os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
import tqdm


class DLProgress(tqdm.tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def gen_batch_function(data_folder, image_shape, 
    maxdata='all', num_classes=2, 
    sample_aug_folder=None, num_aug_sample=10):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """

    class GetBatchesFn(object):
        def __init__(self):
            self.image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
            if isinstance(maxdata, int):
                self.image_paths = self.image_paths[:maxdata]

        def nbatches(self, batch_size):
            return len(range(0, len(self.image_paths), batch_size))

        def augment(self, image, label,
            aug=.1, flip=.5, noise=.5, noiselevel=.01,
            rotrange=(-10., 10.), 
            shiftranges=((-20, 20), (-20, 20))
            ):
            import scipy.ndimage.interpolation
            chk = lambda p: np.random.rand() < p

            # Maybe short-circuit and do nothing?
            if not chk(aug):
                return image, label

            # Maybe flip the image.
            if chk(flip):
                image = np.fliplr(image)
                label = np.fliplr(label)

            # Rotate a little or a lot.
            # Need to use 0-order spline for labels to prevent aliasing phenomenon.
            angle = np.random.uniform(low=rotrange[0], high=rotrange[1])
            image = scipy.ndimage.interpolation.rotate(image, angle, mode='nearest', reshape=False)
            label = scipy.ndimage.interpolation.rotate(label, angle, mode='nearest', reshape=False, order=0)

            # Shift a little or a lot.
            # Need to use 0-order spline for labels to prevent aliasing phenomenon.
            shiftr = np.random.uniform(low=shiftranges[0][0], high=shiftranges[0][1])
            shiftc = np.random.uniform(low=shiftranges[1][0], high=shiftranges[1][1])
            image = scipy.ndimage.interpolation.shift(image, [shiftr, shiftc, 0], mode='nearest',)
            label = scipy.ndimage.interpolation.shift(label, [shiftr, shiftc, 0], mode='nearest', order=0)

            # Maybe add salt-and-PEPPER noise.
            if chk(noise):
                x = np.random.uniform(size=image.shape[:2])
                image[x < noiselevel] = (0, 0, 0)
                image[x > (1 - noiselevel)] = (255, 255, 255)
            
            if not hasattr(self, 'num_sample_aug_saved'): self.num_sample_aug_saved = 0
            if sample_aug_folder and self.num_sample_aug_saved < num_aug_sample:
                t = time.time()
                try:
                    os.makedirs(sample_aug_folder)
                except FileExistsError:
                    pass
                scipy.misc.imsave(os.path.join(sample_aug_folder, '%s_image.png' % t), 
                    image)
                scipy.misc.imsave(os.path.join(sample_aug_folder, '%s_label.png' % t), 
                    label[:, :, 1].astype('uint8') * 255)
                self.num_sample_aug_saved += 1

            return image, label


        def __call__(self, batch_size):
            """
            Create batches of training data
            :param batch_size: Batch Size
            :return: Batches of training data
            """
            
            label_paths = {
                re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
                for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
            background_color = np.array([255, 0, 0])
            foreground_color = np.array([255, 0, 255])

            random.shuffle(self.image_paths)
            for batch_i in range(0, len(self.image_paths), batch_size):
                images = []
                gt_images = []
                for image_file in self.image_paths[batch_i:batch_i+batch_size]:
                    gt_image_file = label_paths[os.path.basename(image_file)]

                    image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                    gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                    if num_classes == 2:
                        # Binary bg/not-bg classification
                        gt_fg = np.all(gt_image == foreground_color, axis=2)
                        gt_fg = gt_fg.reshape(*gt_fg.shape, 1)
                        gt_image = np.concatenate((np.invert(gt_fg), gt_fg), axis=2)
                        gt_images.append(gt_image)

                    else:
                        assert num_classes == 3

                        # Multiple classes.
                        class_colors = np.array([
                            [255, 0, 0],  # Red
                            [255, 0, 255], # Magenta
                            [0, 0, 0]  # Black
                        ])
                        one_hot_image = np.zeros((gt_image.shape[0], gt_image.shape[1], len(class_colors)))
                        for i in range(gt_image.shape[0]):
                            for j in range(gt_image.shape[1]):
                                pixel = gt_image[i, j, :]
                                disparities = [
                                    np.linalg.norm(pixel - color)
                                    for color in class_colors
                                ]
                                closest = np.argmin(disparities)
                                one_hot_image[i, j, closest] = 1
                        gt_images.append(one_hot_image.astype(bool))

                    images.append(image)

                # With small some probability, randomly augment the images.
                for i in range(len(images)):
                    images[i], gt_images[i] = self.augment(images[i], gt_images[i])

                yield np.array(images), np.array(gt_images)
    return GetBatchesFn()


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape, maxdata=20):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    paths = sorted(glob(os.path.join(data_folder, 'image_2', '*.png')))[:maxdata]
    for image_file in tqdm.tqdm(paths, desc=os.path.basename(os.path.normpath(data_folder))):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(
    runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, 
    tag=None, folders=['training', 'testing', 'video', 'video2'], maxdata=20,
    ):
    from os import system
    # Make folder for current run
    if tag is None: tag = str(time.time())
    output_dir = os.path.join(runs_dir, tag)
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass

    # Run NN on test images and save them to HD
    for folder in folders:
        image_outputs = gen_test_output(
            sess, logits, keep_prob, input_image, 
            os.path.join(data_dir, 'data_road', folder), 
            image_shape,
            maxdata=maxdata
            )
        os.makedirs(os.path.join(output_dir, folder))
        for name, image in image_outputs:
            if folder == 'training':
                system('cp  "%s" "%s"' % (
                    os.path.join(data_dir, 'data_road', folder, 'gt_image_2', name.replace('_', '_road_')),
                    os.path.join(output_dir, folder, name.replace('.png', '') + '_original.png')
                ))
            scipy.misc.imsave(os.path.join(output_dir, folder, name), image)
    return output_dir
