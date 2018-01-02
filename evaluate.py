from __future__ import print_function
import os
# import sys
import time

# from PIL import Image
import numpy as np
import tensorflow as tf

# from tools import decode_labels
from image_reader import ImageReader
from mobilenet import MobileNet

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

tf.app.flags.DEFINE_boolean('flipped_eval', True,
                            'whether to evaluate with flipped img.')

tf.app.flags.DEFINE_boolean('print_each_step', True,
                            'whether to print after eah step')

tf.app.flags.DEFINE_integer('gpu', 0,
                            'Which GPU to use.')

tf.app.flags.DEFINE_integer('image_width', 713,
                            'Which GPU to use.')

tf.app.flags.DEFINE_integer('image_height', 713,
                            'Which GPU to use.')

tf.app.flags.DEFINE_integer('num_steps', 500,
                            'No. of images in val dataset')

tf.app.flags.DEFINE_string(
    'evaluate_log_file', 'evaluate.log',
    'File where the results are appended.')


FLAGS = tf.app.flags.FLAGS

FLAGS.num_classes = 19
FLAGS.ignore_label = 255 # Don't care label

#Set Visible CUDA Devices
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(FLAGS.gpu)

def load(sess, file, restore_var):
    loader = tf.train.Saver(var_list=restore_var)
    loader.restore(sess, file)
    print("Restored model parameters from {}".format(file))

def writeToLogFile(epoch,mIoU):
    file = open(FLAGS.evaluate_log_file, 'a')
    s = 'Epoch {0}: {1} mIoU\n'.format(epoch, mIoU)
    file.write(s)
    file.close()


def main():

    temp_flags = FLAGS.__flags.items()
    temp_flags.sort()
    for params, value in FLAGS.__flags.items():
        print('{}: {}'.format(params,value))

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

    print(image_batch)
    net = MobileNet(image_batch, isTraining=False, updateBeta=False)

    with tf.variable_scope('', reuse=True):
        flipped_img = tf.image.flip_left_right(image)
        flipped_img = tf.expand_dims(flipped_img, dim=0)
        net2 = MobileNet(flipped_img, isTraining=False, updateBeta=False)

    # Which variables to load.
    restore_var = tf.global_variables()

    raw_output = net

    if FLAGS.flipped_eval:
        flipped_output = tf.image.flip_left_right(tf.squeeze(net2))
        flipped_output = tf.expand_dims(flipped_output, dim=0)
        raw_output = tf.add_n([raw_output, flipped_output])

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

    load(sess, FLAGS.checkpoint_path, restore_var)

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for step in range(1,FLAGS.num_steps+1):
        preds, _ = sess.run([pred, update_op])

        # if step > 0 and FLAGS.measure_time:
        #     calculate_time(sess, net)

        if FLAGS.print_each_step and step % 100 == 0:
            print('Finish {0}/{1}'.format(step, FLAGS.num_steps))
            print('step {0} mIoU: {1}'.format(step, sess.run(mIoU)))

    value = sess.run(mIoU)
    print('step {0} mIoU: {1}'.format(step, value))

    epoch = int(os.path.basename(FLAGS.checkpoint_path).split('-')[1])
    writeToLogFile(epoch,value)

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    main()
