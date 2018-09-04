# Semantic Segmentation
## Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

![sample classification](sample.png)
![longer animation](anim-video.gif)

Additionally short videos 
[here](https://youtu.be/GdW_vgUg1YA),

[here](https://youtu.be/cZv4Ccd8I8M),
[here](https://youtu.be/0fpr8EizK7Y),
and
[here](https://youtu.be/Uw1aytYEH6E).


### Submission
☒ Ensure you've passed all the unit tests.

☒ Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).

☐ Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.

## Training Data

The dataset contains images with filenames of the form `*_lane_*` and `*_road_*` -- the former mark just the ego-lane in which the viewpoing car is traveling, while the latter mark the whole drivable surface of the road.
![Sample image](doc/lane_vs_road/um_000062.png)
![lane GT](doc/lane_vs_road/um_lane_000062.png)
![road GT](doc/lane_vs_road/um_road_000062.png)
To simplify the problem, I used only data of the second form.

Additionally, some parts of the dataset split the GT image into three classes rather than two, being the lane or road on our side of a highway divider, the road on the other side, and other non-road features. At one point, I briefly tried to train such a three-class segmenter, but stopped quickly due to the large increase in trainable parameters and poor initial training results. After this, I merged the opposing-traffic class and the offroad-class into one.

## Network Architecture

The architecture for this project follows Long et al. [1].

[![](doc/graph-1535478125.5632854.png)](doc/graph-1535478125.5632854.pdf "network architecture diagram (click for PDF)")

In summary, we remove the fully-connected classifier portion from a pre-trained VGG16 network, take three intermediate feature-map layers from the network of differing sizes (in width, height, and number of channels), and combine these, upscaling as appropriate. Upscaling is performed as a so-called transposed convolution, also know as a fractionally-strided convolution or a deconvolution. Combination is performed with addition, but, per [1], this is equivalent to concatenation of features via the linearity of the convolutions--no ReLU or other nonlinearities are added in this post-VGG16 "decoder" section of the network.

The deconvolutions serve to align the feature maps from multiple scales in width and height. However, before combining features by addition, we use a 1x1 convolution to produce our target number-of-features as a linear combination of the 256 or 512 features provided by two different layers of VGG16. This is equivalent to using a fully-connected layer pointwise across the respective feature maps, but ensures that there are no fixed-size layers in the entire network (the network is "fully convolutional"), and the output will always have the same width and height as the input image, allowing us to apply it to images of arbitrary size (above a certain minimum).

The combination of information from multiply-scaled feature maps is called a "skip connection", in reference to the structure of the figure above. Again, per [1], these skip connections are initialized with an all-zeros 1x1 convolution weight matrix. That is, at start of training, only information from the coarsest level of the VGG16 encoder reaches the output. Additionally, features from higher-resolution layers in the encoder are multiplied by progressively smaller scaling factors before combining with lower-resolution features, to prevent the higher from dominating the lower.

The upscaling deconvolutions were initialized as bilinear interpolation, though they were permitted to vary during training.

All of these small details, suggested by the Long paper, did not visually add to or detract from the quality of the resulting videos (except for the bilinear interpolation initialization, which helped considerably over Glorot/Xavier initialization). However, one addition that made a really big difference was data augmentation. To help with generalization, I supplemented the training dataset with image inputs corrupted with random horizontal flips, small rotations, small translations, and salt-and-pepper noise (some pixels randomly set to black or white).

Before data agumentation:
![c92c873](doc/08-c92c873.png)

After data augmentation:
![fe5e2e7](doc/14-fe5e2e7.png)

Besides data augmentation, the only other thing that seemed to provide major improvements in visual quality was long training at a moderate to low learning rate. However, I additionally tried several other tricks, with some small improvement possibly attributed to each. Namely: (1) First, I trained the whole network (VGG16 layers and the new decoder layers), then I restricted the second half of the training to only the decoder layers. (2) I used a learning-rate decay schedule (annealing), where the learning rate decayed exponentially over up to two orders of magnitude during about an hour and a half of training.


## Results

For the most part, the network is able to identify drivable space fairly well.

![loss history](doc/losshist.png)

![animation](anim.gif)

![second animation](anim2.gif)

![anim-2011_09_26_drive_0048_sync](doc/anim-2011_09_26_drive_0048_sync.gif)

![anim-2011_09_26_drive_0051_sync](doc/anim-2011_09_26_drive_0051_sync.gif)

However, it still has trouble with very high-contrast and sharp shadow edges


![anim-2011_09_26_drive_0117_sync](doc/anim-2011_09_26_drive_0117_sync.gif)

and also scenes where I as a human am confused about where exactly to draw the road edge (and so where the human labelers may have been inconsistent as well)

![anim-2011_09_26_drive_0091_sync](doc/anim-2011_09_26_drive_0091_sync.gif)

My next step will be to gather a similar dataset from the point of view of an RC car driving around an office, retrain on this network, and then used this to identify drivable space for a path planner, in a manner similar to the LAGR rover [2,3] developed by Yann LeCun and colleagues at NYU in the previous decade, though I'll have Ackermann kinematics rather than differential drive.

## Some commit messages.

#### Good further changes

    28. Do full training first.

    29. Do twice as many epochs

    30. Decay the learning rate.

    31. Use a less-aggressive decay schedule.



#### Bad further changes

    Do L<sub>2</sub> regularization manually.

    Try a very small L<sub>2</sub> parameter.

    Try not training the VGG16 layers.

    Try a larger L<sub>2</sub> with frozen VGG16.

    Train first just decoder then whole net

    Decay the learning rate.

    Start with a larger LR.

    Larger LR+smaller (stronger) decay.



## References

[1] J. Long, E. Shelhamer, and T. Darrell, "Fully convolutional networks for semantic segmentation,"" in Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2015, vol. 07-12-June-2015.

[2] ["LAGR: Learning Applied to Ground Robotics"](https://cs.nyu.edu/~yann/research/lagr/#videos)

[3] R. Hadsell et al., ["Online Learning for Offroad Robots: Using Spatial Label Propagation to Learn Long-Range Traversability."](https://cs.nyu.edu/~yann/research/lagr/#papers)
