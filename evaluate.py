from __future__ import print_function
import os
import sys
import time

from PIL import Image
import tensorflow as tf
import numpy as np

from tools import decode_labels
from image_reader import ImageReader

slim = tf.contrib.slim

IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)
input_size = [1024, 2048]

tf.app.flags.DEFINE_string(
    'data_dir', '/home/n1703300e/SS/Datasets/cityscapes-images/',
    'Directory where the data is located.')

tf.app.flags.DEFINE_string(
    'data_list', 'list/eval_list.txt',
    'Path to file where the image list is stored.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '',
    'Directory where the data is located.')

tf.app.flags.DEFINE_string(
    'save_dir', '',
    'Directory where the data is located.')

tf.app.flags.DEFINE_boolean('flipped-eval', True,
                            'whether to evaluate with flipped img.')

tf.app.flags.DEFINE_integer('gpu', 0,
                            'Which GPU to use.')

tf.app.flags.DEFINE_boolean('print_architecture', False,
                            'Print architecure.')

tf.app.flags.DEFINE_integer('image_width', 2048,
                            'Which GPU to use.')

tf.app.flags.DEFINE_integer('image_height', 1024,
                            'Which GPU to use.')

tf.app.flags.DEFINE_integer('num_steps', 500,
                            'No. of images in train dataset')


FLAGS = tf.app.flags.FLAGS

FLAGS.num_classes = 19
FLAGS.ignore_label = 255 # Don't care label

#Set Visible CUDA Devices
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(FLAGS.gpu)

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

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    print(FLAGS)

    coord = tf.train.Coordinator()
    tf.reset_default_graph()

    input_size = [FLAGS.image_height, FLAGS.image_width]

    reader = ImageReader(
            FLAGS.data_dir,
            FLAGS.data_list,
            input_size,
            None,
            None,
            FLAGS.ignore_label,
            IMG_MEAN,
            coord)
    image, label = reader.image, reader.label

    image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0) # Add one batch dimension.

    net = mobilenet(image_batch, is_training = False)
    #TODO Add flipped eval

    # Which variables to load.
    restore_var = tf.global_variables()

    raw_output = net

    raw_output_up = tf.image.resize_bilinear(raw_output, size=input_size, align_corners=True)
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    # mIoU
    pred_flatten = tf.reshape(pred, [-1,])
    raw_gt = tf.reshape(label_batch, [-1,])
    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, FLAGS.num_classes - 1)), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    pred = tf.gather(pred_flatten, indices)

    mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=FLAGS.num_classes)

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)
    sess.run(tf.local_variables_initializer())

    restore_var = tf.global_variables()

    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found.')

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for step in range(1,FLAGS.num_steps+1):
        preds, _ = sess.run([pred, update_op])

        # if step > 0 and FLAGS.measure_time:
        #     calculate_time(sess, net)

        if step % 10 == 0:
            print('Finish {0}/{1}'.format(step, FLAGS.num_steps))
            print('step {0} mIoU: {1}'.format(step, sess.run(mIoU)))

    print('step {0} mIoU: {1}'.format(step, sess.run(mIoU)))

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    main()
