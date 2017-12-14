from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from image_reader import ImageReader
from tools import decode_labels, prepare_label

import time
import os
import os.path as osp
import sys
import datetime
from PIL import Image
import numpy as np
np.set_printoptions(threshold=np.nan)
from scipy import misc
import matplotlib.pyplot as plt

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'data_dir', '/home/n1703300e/SS/Datasets/cityscapes-images/',
    'Directory where the data is located.')

tf.app.flags.DEFINE_string(
    'data_list', 'list/train_list.txt',
    'Path to file where the image list is stored.')

tf.app.flags.DEFINE_string(
    'log_dir', 'logs/train1_FINE-FULL-SGD/',
    'Directory where the data is located.')

tf.app.flags.DEFINE_string(
    'pretrained_check_point', 'MobileNetPreTrained/model.ckpt-906808',
    'Directory where the data is located.')

tf.app.flags.DEFINE_boolean('random_scale', True,
                            'Whether to randomly scale the inputs during the training.')

tf.app.flags.DEFINE_boolean('random_mirror', True,
                            'Whether to randomly scale the inputs during the training.')

tf.app.flags.DEFINE_boolean('ignore_label', True,
                            'The index of the label to ignore during the training.')


tf.app.flags.DEFINE_integer('gpu', 0,
                            'Which GPU to use.')


tf.app.flags.DEFINE_boolean('print_architecture', True,
                            'Print architecure.')

tf.app.flags.DEFINE_boolean('print_info', False,
                            'Print info.')

tf.app.flags.DEFINE_boolean('do_validaiton', True,
                            'Perform validation')

tf.app.flags.DEFINE_boolean('use_latest_weights', False,
                            'Use latest weights.')

tf.app.flags.DEFINE_integer('image_width', 2048,
                            'Which GPU to use.')

tf.app.flags.DEFINE_integer('image_height', 1024,
                            'Which GPU to use.')

tf.app.flags.DEFINE_integer('num_readers', 8,
                            'Which GPU to use.')

tf.app.flags.DEFINE_integer('num_preprocessing_threads', 8,
                            'Which GPU to use.')

tf.app.flags.DEFINE_integer('batch_size', 1,
                            'Which GPU to use.')

tf.app.flags.DEFINE_integer('num_classes', 19,
                            'Which GPU to use.')

tf.app.flags.DEFINE_integer('train_image_size', 713,
                            'Which GPU to use.')

tf.app.flags.DEFINE_integer('start_epoch', 1,
                            'Which GPU to use.')

tf.app.flags.DEFINE_integer('end_epoch', 200,
                            'Which GPU to use.')

tf.app.flags.DEFINE_string('optimizer', 'sgd',
                            'Which GPU to use.')

tf.app.flags.DEFINE_integer('start_learning_rate', 0.001,
                            'Which GPU to use.')

tf.app.flags.DEFINE_integer('end_learning_rate', 0.00001,
                            'Which GPU to use.')

tf.app.flags.DEFINE_integer('decay_steps', 20,
                            'Which GPU to use.')

tf.app.flags.DEFINE_integer('learning_rate_decay_power', 1,
                            'Which GPU to use.')

tf.app.flags.DEFINE_integer('learning_rate_decay_factor', 0.5,
                            'Which GPU to use.')

tf.app.flags.DEFINE_integer('weight_decay', 0.5,
                            'Which GPU to use.')

tf.app.flags.DEFINE_float('momentum', 0.9,
                          '')


FLAGS = tf.app.flags.FLAGS
FLAGS.my_pretrained_weights = FLAGS.log_dir
FLAGS.num_epochs = FLAGS.end_epoch - FLAGS.start_epoch + 1

IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)

sys.path.insert(0, FLAGS.data_dir+'cityscapesScripts/cityscapesscripts/helpers')
from labels import id2label, trainId2label

#Set Visible CUDA Devices
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(FLAGS.gpu)


#Change these files to change the Dataset, 'train_fine.tfRecord' for Fine Dataset / 'train_coarse.tfRecord' for Coarse Dataset and same for val.
#TODO Settle Fine Coarse
# dataset_filenames = {
#     'train': 'train_fine.tfRecord',
#     'val': 'val_fine.tfRecord'
# }

#TODO Settle Fine Coarse Number
#Number of images in dataset
# NUM_SAMPLES = {
#         'train': 2975,
#         'val': 500,
# }

# _ITEMS_TO_DESCRIPTIONS = {
#             'image_height' : 'height',
#             'image_width' : 'width',
#             'image_filename' : 'filename',
#             'label': 'label',
#             'image': 'A color image of varying height and width',
# }

summaries_every_iter = []
summaries_costly = []
summaries_images = []
summaries_images_val=[]
TrainStep = tf.placeholder(tf.bool)
train_mean_loss_summary = []
val_mean_loss_summary = []

def print_info():
    print('Start Epoch: %d' % (FLAGS.start_epoch))
    print('End Epoch: %d' % (FLAGS.end_epoch))
    print('No. Of Epochs: %d' % (FLAGS.num_epochs))
    print('Total Running Steps: %d\n' % (FLAGS.num_epochs*NUM_SAMPLES['train']))
    print('Batch Size: %d' % (FLAGS.batch_size))
    print('Train Image Size: %d' % (FLAGS.train_image_size))
    print('No. Of Classes: %d\n' % (FLAGS.num_classes))
    print('Optimizer: %s' % (FLAGS.optimizer))
    print('Start learning Rate: %f' % (FLAGS.start_learning_rate))
    print('End learning Rate: ')
    print('Learning Rate Decay Factor: %f' % (FLAGS.learning_rate_decay_factor))
    print('Decay Learning Rate After %d Epoch\n' % (FLAGS.num_epochs_per_delay))


def mobilenet(inputs, num_classes=19, is_training=True, width_multiplier=1, scope='MobileNet'):

    def _depthwise_separable_conv(inputs, num_pwc_filters, width_multiplier, sc, downsample=False):
        num_pwc_filters = round(num_pwc_filters * width_multiplier)
        _stride = 2 if downsample else 1

        depthwise_conv = slim.separable_convolution2d(inputs, num_outputs=None, stride=_stride, depth_multiplier=1, kernel_size=[3, 3], scope=sc+'/depthwise_conv')
        bn = slim.batch_norm(depthwise_conv, scope=sc+'/dw_batch_norm')
        pointwise_conv = slim.convolution2d(bn, num_pwc_filters, kernel_size=[1, 1], scope=sc+'/pointwise_conv')
        bn = slim.batch_norm(pointwise_conv, scope=sc+'/pw_batch_norm')
        return bn

    def _depthwise_separable_convPSP(inputs, kernel, stride, num_pwc_filters, width_multiplier, sc):
        num_pwc_filters = round(num_pwc_filters * width_multiplier)

        # depthwise_conv = slim.separable_convolution2d(inputs, num_outputs=None, stride=stride, depth_multiplier=1, kernel_size=kernel, scope=sc+'/depthwise_conv')
        # bn = slim.batch_norm(depthwise_conv, scope=sc+'/dw_batch_norm')

        conv_layer = slim.convolution2d(inputs, num_pwc_filters, kernel_size=kernel, stride=stride, scope=sc+'/conv')
        bn = slim.batch_norm(conv_layer, scope=sc+'/batch_norm')

        # pointwise_conv = slim.convolution2d(bn, num_pwc_filters, kernel_size=[1, 1], scope=sc+'/pointwise_conv')
        # bn = slim.batch_norm(pointwise_conv, scope=sc+'/pw_batch_norm')
        return bn

    def _pointwisePSP(inputs, num_pwc_filters, width_multiplier, sc):
        num_pwc_filters = round(num_pwc_filters * width_multiplier)
        pointwise_conv = slim.convolution2d(inputs, num_pwc_filters, kernel_size=[1, 1], scope=sc+'/pointwise_conv')
        bn = slim.batch_norm(pointwise_conv, scope=sc+'/pw_batch_norm')
        return bn

    with slim.arg_scope( [slim.convolution2d, slim.separable_convolution2d],
                                weights_initializer=slim.initializers.xavier_initializer(),
                                biases_initializer=slim.init_ops.zeros_initializer()):
        with tf.variable_scope(scope) as sc:
            with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d], activation_fn=None):
                with slim.arg_scope([slim.batch_norm], is_training=is_training, activation_fn=tf.nn.relu, fused=False):
                    net = slim.convolution2d(inputs, round(32 * width_multiplier), [3, 3], stride=2, padding='SAME', scope='conv_1')
                    net = slim.batch_norm(net, scope='conv_1/batch_norm')
                    if FLAGS.print_architecture: print('after conv_1: ',net)
                    net = _depthwise_separable_conv(net, 64, width_multiplier, sc='conv_ds_2')
                    if FLAGS.print_architecture: print('after conv_ds_2: ',net)
                    net = _depthwise_separable_conv(net, 128, width_multiplier, downsample=True, sc='conv_ds_3')
                    if FLAGS.print_architecture: print('after conv_ds_3: ',net)
                    net = _depthwise_separable_conv(net, 128, width_multiplier, sc='conv_ds_4')
                    if FLAGS.print_architecture: print('after conv_ds_4: ',net)
                    net = _depthwise_separable_conv(net, 256, width_multiplier, downsample=True, sc='conv_ds_5')
                    if FLAGS.print_architecture: print('after conv_ds_5: ',net)
                    net = _depthwise_separable_conv(net, 256, width_multiplier, sc='conv_ds_6')
                    if FLAGS.print_architecture: print('after conv_ds_6: ',net)
                    net = _depthwise_separable_conv(net, 512, width_multiplier, downsample=False, sc='conv_ds_7')
                    if FLAGS.print_architecture: print('after conv_ds_7: ',net)
                    net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_8')
                    if FLAGS.print_architecture: print('after conv_ds_8: ',net)
                    net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_9')
                    if FLAGS.print_architecture: print('after conv_ds_9: ',net)
                    net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_10')
                    if FLAGS.print_architecture: print('after conv_ds_10: ',net)
                    net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_11')
                    if FLAGS.print_architecture: print('after conv_ds_11: ',net)
                    net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_12')
                    if FLAGS.print_architecture: print('after conv_ds_12: ',net)
                    net = _depthwise_separable_conv(net, 1024, width_multiplier, downsample=False, sc='conv_ds_13')
                    if FLAGS.print_architecture: print('after conv_ds_13: ',net)
                    net = _depthwise_separable_conv(net, 1024, width_multiplier, sc='conv_ds_14')
                    if FLAGS.print_architecture: print('after conv_ds_14: ',net)

                    net_a = slim.avg_pool2d(net, [90,90], stride=90, scope='conv_ds_15a/pool_15_1a')
                    if FLAGS.print_architecture: print('after pool_15_1a: ',net_a)
                    net_b = slim.avg_pool2d(net, [45,45], stride=45, scope='conv_ds_15b/pool_15_1b')
                    if FLAGS.print_architecture: print('after pool_15_1b: ',net_b)
                    net_c = slim.avg_pool2d(net, [30,30], stride=30, scope='conv_ds_15c/pool_15_1c')
                    if FLAGS.print_architecture: print('after pool_15_1c: ',net_c)
                    net_d = slim.avg_pool2d(net, [15,15], stride=15, scope='conv_ds_15d/pool_15_1d')
                    if FLAGS.print_architecture: print('after pool_15_1d: ',net_d)

                    net_a = _pointwisePSP(net_a, 256, width_multiplier, sc='conv_ds_15a/conv_ds_15_2a')
                    if FLAGS.print_architecture: print('after conv_ds_15_2a: ',net_a)
                    net_b= _pointwisePSP(net_b,  256, width_multiplier, sc='conv_ds_15b/conv_ds_15_2b')
                    if FLAGS.print_architecture: print('after conv_ds_15_2b: ',net_b)
                    net_c = _pointwisePSP(net_c, 256, width_multiplier, sc='conv_ds_15c/conv_ds_15_2c')
                    if FLAGS.print_architecture: print('after conv_ds_15_2c: ',net_c)
                    net_d = _pointwisePSP(net_d, 256, width_multiplier, sc='conv_ds_15d/conv_ds_15_2d')
                    if FLAGS.print_architecture: print('after conv_ds_15_2d: ',net_d)

                    net_a = tf.image.resize_bilinear(net_a, [90,90], align_corners=True, name='conv_ds_15a/conv_t1')
                    net_b = tf.image.resize_bilinear(net_b, [90,90], align_corners=True, name='conv_ds_15b/conv_t1')
                    net_c = tf.image.resize_bilinear(net_c, [90,90], align_corners=True, name='conv_ds_15c/conv_t1')
                    net_d = tf.image.resize_bilinear(net_d, [90,90], align_corners=True, name='conv_ds_15d/conv_t1')

                    fuse_15 = tf.concat([net, net_a,net_b,net_c,net_d],axis=3, name='fuse_psp')
                    if FLAGS.print_architecture: print('after fuse_15: ',fuse_15)

                    net = _depthwise_separable_convPSP(fuse_15, [3,3], 1, 256, width_multiplier, sc='conv_ds_16')
                    if FLAGS.print_architecture: print('after conv_ds_16: ',net)
                    net = _depthwise_separable_convPSP(net, [3,3], 1, 19, width_multiplier, sc='conv_ds_17')
                    if FLAGS.print_architecture: print('after conv_ds_17: ',net)

            # annotation_pred = tf.argmax(net, dimension=3, name="prediction")
    return net
    # return tf.cast(tf.expand_dims(annotation_pred, dim=3),dtype=tf.uint8), net



def weights_initialisers():
    if FLAGS.use_latest_weights:
        restoreAllVars = slim.get_variables_to_restore()
        print(
            'Ignoring %s because a checkpoint already exists in %s'
             % (FLAGS.pretrained_check_point, FLAGS.my_pretrained_weights) )
        checkpoint_path = FLAGS.my_pretrained_weights
        if tf.gfile.IsDirectory(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        return slim.assign_from_checkpoint_fn(checkpoint_path,restoreAllVars)

    restoreVar_mobilenet = slim.get_variables_to_restore(include=['MobileNet'],exclude=['MobileNet/conv_ds_15a','MobileNet/conv_ds_15b','MobileNet/conv_ds_15c','MobileNet/conv_ds_15d','MobileNet/conv_ds_16','MobileNet/conv_ds_17'])
    restoreVar_mobilenet = [v for v in restoreVar_mobilenet if 'Momentum' not in v.name]
    newLayerVariables = slim.get_variables_to_restore(include=['MobileNet/conv_ds_15a','MobileNet/conv_ds_15b','MobileNet/conv_ds_15c','MobileNet/conv_ds_15d','MobileNet/conv_ds_16','MobileNet/conv_ds_17'])
    optimizer_variables = slim.get_variables_to_restore(exclude=['MobileNet'])

    checkpoint_path=FLAGS.pretrained_check_point

    print('Restoring weights from  %s: ' % (checkpoint_path) )
    readMobileNetWeights = slim.assign_from_checkpoint_fn(checkpoint_path,restoreVar_mobilenet)
    otherLayerInitializer = tf.variables_initializer(newLayerVariables)
    restInitializer = tf.variables_initializer(optimizer_variables)
    return readMobileNetWeights, otherLayerInitializer, restInitializer

def main():

    if FLAGS.print_info:
        print_info()

    input_size = (FLAGS.train_image_size, FLAGS.train_image_size)

    tf.set_random_seed(1234)

    coord = tf.train.Coordinator()

    reader = ImageReader(
            FLAGS.data_dir,
            FLAGS.data_list,
            input_size,
            FLAGS.random_scale,
            FLAGS.random_mirror,
            FLAGS.ignore_label,
            IMG_MEAN,
            coord)
    image_batch, label_batch = reader.dequeue(FLAGS.batch_size)

    raw_output = mobilenet(image_batch)

    psp_list = ['conv_ds_15a','conv_ds_15b','conv_ds_15c','conv_ds_15d','conv_ds_16','conv_ds_17']
    all_trainable = [v for v in tf.trainable_variables()]
    psp_trainable = [v for v in all_trainable if v.name.split('/')[1] in psp_list and ('weights' in v.name or 'biases' in v.name)]
    conv_trainable = [v for v in all_trainable if v not in psp_trainable] # lr * 1.0
    psp_w_trainable = [v for v in psp_trainable if 'weights' in v.name] # lr * 10.0
    psp_b_trainable = [v for v in psp_trainable if 'biases' in v.name] # lr * 20.0

    assert(len(all_trainable) == len(psp_trainable) + len(conv_trainable))
    assert(len(psp_trainable) == len(psp_w_trainable) + len(psp_b_trainable))

    # Predictions: ignoring all pall_trainable = [v for v in tf.trainable_variables()]redictions with labels greater or equal than n_classes
    raw_prediction = tf.reshape(raw_output, [-1, FLAGS.num_classes])
    label_proc = prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]), num_classes=FLAGS.num_classes, one_hot=False) # [batch_size, h, w]
    raw_gt = tf.reshape(label_proc, [-1,])
    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, FLAGS.num_classes - 1)), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    prediction = tf.gather(raw_prediction, indices)

    # Pixel-wise softmax loss.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
    l2_losses = [FLAGS.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)
    #TODO  auxilary loss

    #Using Poly learning rate policy
    current_epoch = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.train.polynomial_decay(FLAGS.start_learning_rate, current_epoch, FLAGS.decay_steps, end_learning_rate=FLAGS.end_learning_rate, power=FLAGS.learning_rate_decay_power, name="poly_learning_rate")

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        opt_conv = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
        opt_psp_w = tf.train.MomentumOptimizer(learning_rate * 10.0, FLAGS.momentum)
        opt_psp_b = tf.train.MomentumOptimizer(learning_rate * 20.0, FLAGS.momentum)

        grads = tf.gradients(reduced_loss, conv_trainable + psp_w_trainable + psp_b_trainable)
        grads_conv = grads[:len(conv_trainable)]
        grads_psp_w = grads[len(conv_trainable) : (len(conv_trainable) + len(psp_w_trainable))]
        grads_psp_b = grads[(len(conv_trainable) + len(psp_w_trainable)):]

        train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable))
        train_op_psp_w = opt_psp_w.apply_gradients(zip(grads_psp_w, psp_w_trainable))
        train_op_psp_b = opt_psp_b.apply_gradients(zip(grads_psp_b, psp_b_trainable))

        train_op = tf.group(train_op_conv, train_op_psp_w, train_op_psp_b)

    # if FLAGS.use_latest_weights:
    #     MobileNetAllWeightsFunction = weights_initialisers()
    # else:
    #     MobileNetWeightsFunction, otherLayersInitializer, restInitializer =  weights_initialisers()
    # localvariables = tf.initialize_local_variables()

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.initialize_all_variables()

    if FLAGS.use_latest_weights:
	 MobileNetAllWeightsFunction = weights_initialisers()
    else:
	MobileNetWeightsFunction, otherLayersInitializer, restInitializer =  weights_initialisers()
    localvariables = tf.initialize_local_variables()



    sess.run(init)
    if FLAGS.use_latest_weights:
	MobileNetAllWeightsFunction(sess)
    else:
	MobileNetWeightsFunction(sess)
	sess.run(otherLayersInitializer)		
	sess.run(restInitializer)
    sess.run(localvariables)

    load_step = 0

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

# Iterate over training steps.
    for step in range(1000):
        start_time = time.time()

        feed_dict = {current_epoch: 0}
        # if step % args.save_pred_every == 0:
            # loss_value, _ = sess.run([reduced_loss, train_op], feed_dict=feed_dict)
            # save(saver, sess, args.snapshot_dir, step)
        # else:
        loss_value, _ = sess.run([reduced_loss, train_op], feed_dict=feed_dict)
        duration = time.time() - start_time
        print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))


    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    main()
