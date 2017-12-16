from __future__ import print_function

import numpy as np
import tensorflow as tf

from image_reader import ImageReader
from tools import decode_labels, prepare_label
from mobilenet import mobilenet

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

tf.app.flags.DEFINE_boolean('random_scale', False,
                            'Whether to randomly scale the inputs during the training.')

tf.app.flags.DEFINE_boolean('random_mirror', True,
                            'Whether to randomly scale the inputs during the training.')

tf.app.flags.DEFINE_integer('ignore_label', 255,
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

tf.app.flags.DEFINE_integer('num_epochs', 50,
                            'Which GPU to use.')

tf.app.flags.DEFINE_integer('num_steps', 2975,
                            'No. of images in train dataset')

# tf.app.flags.DEFINE_integer('end_epoch', 200,
#                             'Which GPU to use.')

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
# FLAGS.num_epochs = FLAGS.end_epoch - FLAGS.start_epoch + 1

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


def weights_initialisers():
    # if FLAGS.use_latest_weights:
    #     restoreAllVars = slim.get_variables_to_restore()
    #     print(
    #         'Ignoring %s because a checkpoint already exists in %s'
    #          % (FLAGS.pretrained_check_point, FLAGS.my_pretrained_weights) )
    #     checkpoint_path = FLAGS.my_pretrained_weights
    #     if tf.gfile.IsDirectory(checkpoint_path):
    #         checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    #     return slim.assign_from_checkpoint_fn(checkpoint_path,restoreAllVars)

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

def save(saver, sess, logdir, step):
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)

   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

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

    raw_output = Mobilenet(image_batch)

    psp_list = ['conv_ds_15a','conv_ds_15b','conv_ds_15c','conv_ds_15d','conv_ds_16','conv_ds_17']
    all_trainable = [v for v in tf.trainable_variables()]
    psp_trainable = [v for v in all_trainable if v.name.split('/')[1] in psp_list and ('weights' in v.name or 'biases' in v.name)]
    conv_trainable = [v for v in all_trainable if v not in psp_trainable] # lr * 1.0
    psp_w_trainable = [v for v in psp_trainable if 'weights' in v.name] # lr * 10.0
    psp_b_trainable = [v for v in psp_trainable if 'biases' in v.name] # lr * 20.0

    restore_var = [v for v in all_trainable if (v.name.split('/')[1] not in psp_list and 'Momentum' not in v.name)]

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

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=100)

    loader = tf.train.Saver(var_list=restore_var)
    loader.restore(sess, FLAGS.pretrained_check_point)

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    for epoch in range(1, FLAGS.num_epochs+1):

        total_loss = 0.0
        for step in range(1,FLAGS.num_steps+1):

            start_time = time.time()

            feed_dict = {current_epoch: epoch}
            loss_value, _ = sess.run([reduced_loss, train_op], feed_dict=feed_dict)

            duration = time.time() - start_time
            print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))

            total_loss += loss
            # if step % args.save_pred_every == 0:
                # loss_value, _ = sess.run([reduced_loss, train_op], feed_dict=feed_dict)
        save(saver, sess, FLAGS.log_dir, epoch)
        total_loss /= FLAGS.num_steps
        print('Epoch {:d} completed! Total Loss = {:.3f}'.format(epoch, total_loss))

            # else:


    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    main()
