from __future__ import print_function

import numpy as np
import tensorflow as tf

from image_reader import ImageReader
from tools import decode_labels, prepare_label
from mobilenet import MobileNet

import time
import os

slim = tf.contrib.slim

#Directory Papths
tf.app.flags.DEFINE_string(
    'data_dir', '/home/n1703300e/SS/Datasets/cityscapes-images/',
    'Directory where the data is located.')

tf.app.flags.DEFINE_string(
    'data_list', 'list/train_list.txt',
    'Path to file where the image list is stored.')

tf.app.flags.DEFINE_string(
    'log_dir', 'logs/train1-Fine-Full-Momentum/',
    'Directory where the data is located.')

tf.app.flags.DEFINE_string(
    'pretrained_checkpoint', '',
    'Directory where the data is located.')

tf.app.flags.DEFINE_boolean('random_scale', True,
                            'Whether to randomly scale the inputs during the training.')

tf.app.flags.DEFINE_boolean('random_mirror', True,
                            'Whether to randomly scale the inputs during the training.')

tf.app.flags.DEFINE_integer('ignore_label', 255,
                            'The index of the label to ignore during the training.')

tf.app.flags.DEFINE_integer('gpu', 0,
                            'Which GPU to use.')

tf.app.flags.DEFINE_integer('batch_size', 1,
                            'Which GPU to use.')

tf.app.flags.DEFINE_integer('num_classes', 19,
                            'Which GPU to use.')

tf.app.flags.DEFINE_integer('train_image_size', 713,
                            'Which GPU to use.')

tf.app.flags.DEFINE_integer('num_epochs', 100,
                            'Which GPU to use.')

tf.app.flags.DEFINE_integer('num_steps', 2975,
                            'No. of images in train dataset')

tf.app.flags.DEFINE_integer('start_epoch', None,
                            'This decides the current learning rate.')

tf.app.flags.DEFINE_integer('start_learning_rate', 0.01,
                            'Which GPU to use.')

tf.app.flags.DEFINE_integer('end_learning_rate', 0.0001,
                            'Which GPU to use.')

tf.app.flags.DEFINE_integer('decay_steps', 40,
                            'Which GPU to use.')

tf.app.flags.DEFINE_integer('learning_rate_decay_power', 1,
                            'Which GPU to use.')

tf.app.flags.DEFINE_string('optimizer', 'momentum',
                            'momentum/rmsprop')

tf.app.flags.DEFINE_float('momentum', 0.9,
                          'momentum for Momentum Optimizer')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9,
                          'Decay term for RMSProp.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9,
                          'momentum for RMSProp Optimizer')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0,
                          'Epsilon term for the optimizer.')


tf.app.flags.DEFINE_integer('weight_decay', 0.00004,
                            'Regularisation Parameter.')

tf.app.flags.DEFINE_boolean('update_beta', True,
                            'Train without changing beta of batch norm layer.')

tf.app.flags.DEFINE_boolean('update_mean_var', True,
                            'whether to get update_op from tf.Graphic_Key.')

FLAGS = tf.app.flags.FLAGS

IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)

#Set Visible CUDA Devices
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(FLAGS.gpu)



def load(sess, file, restore_var):
    loader = tf.train.Saver(var_list=restore_var)
    loader.restore(sess, file)
    print("Restored model parameters from {}".format(file))

def save(saver, sess, logdir, step):
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, 'checkpoints', model_name)

   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def main():

    temp_flags = FLAGS.__flags.items()
    temp_flags.sort()
    for params, value in FLAGS.__flags.items():
        print('{}: {}'.format(params,value))

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

    raw_output = MobileNet(image_batch, isTraining=True, updateBeta=FLAGS.update_beta)

    psp_list = ['conv_ds_15a','conv_ds_15b','conv_ds_15c','conv_ds_15d','conv_ds_16','conv_ds_17']
    all_trainable = [v for v in tf.trainable_variables()]
    if FLAGS.update_beta == False:
        all_trainable = [v for v in all_trainable if 'beta' not in v.name]
    psp_trainable = [v for v in all_trainable if v.name.split('/')[1] in psp_list and ('weights' in v.name or 'biases' in v.name)]
    conv_trainable = [v for v in all_trainable if v not in psp_trainable] # lr * 1.0
    psp_w_trainable = [v for v in psp_trainable if 'weights' in v.name] # lr * 10.0
    psp_b_trainable = [v for v in psp_trainable if 'biases' in v.name] # lr * 20.0

    assert(len(all_trainable) == len(psp_trainable) + len(conv_trainable))
    assert(len(psp_trainable) == len(psp_w_trainable) + len(psp_b_trainable))

    # Predictions: ignoring all predictions with labels greater or equal than n_classes
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

    if FLAGS.update_mean_var == False:
        update_ops = None
    else:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        if FLAGS.optimizer == 'momentum':
            opt_conv = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
            opt_psp_w = tf.train.MomentumOptimizer(learning_rate * 10.0, FLAGS.momentum)
            opt_psp_b = tf.train.MomentumOptimizer(learning_rate * 20.0, FLAGS.momentum)
        elif FLAGS.optimizer == 'rmsprop':
            opt_conv = tf.train.RMSPropOptimizer(learning_rate, decay=FLAGS.rmsprop_decay, momentum=FLAGS.rmsprop_momentum, epsilon=FLAGS.opt_epsilon)
            opt_psp_w = tf.train.RMSPropOptimizer(learning_rate * 10.0, decay=FLAGS.rmsprop_decay, momentum=FLAGS.rmsprop_momentum, epsilon=FLAGS.opt_epsilon)
            opt_psp_b = tf.train.RMSPropOptimizer(learning_rate * 20.0, decay=FLAGS.rmsprop_decay, momentum=FLAGS.rmsprop_momentum, epsilon=FLAGS.opt_epsilon)

        grads = tf.gradients(reduced_loss, conv_trainable + psp_w_trainable + psp_b_trainable)
        grads_conv = grads[:len(conv_trainable)]
        grads_psp_w = grads[len(conv_trainable) : (len(conv_trainable) + len(psp_w_trainable))]
        grads_psp_b = grads[(len(conv_trainable) + len(psp_w_trainable)):]

        train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable))
        train_op_psp_w = opt_psp_w.apply_gradients(zip(grads_psp_w, psp_w_trainable))
        train_op_psp_b = opt_psp_b.apply_gradients(zip(grads_psp_b, psp_b_trainable))

        train_op = tf.group(train_op_conv, train_op_psp_w, train_op_psp_b)

    restore_var = tf.global_variables()

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=500)

    load(sess, FLAGS.pretrained_checkpoint, restore_var)

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    for epoch in range(FLAGS.start_epoch, FLAGS.start_epoch+FLAGS.num_epochs):

        total_loss = 0.0
        for step in range(1,FLAGS.num_steps+1):

            start_time = time.time()

            feed_dict = {current_epoch: epoch}
            loss_value, _ = sess.run([reduced_loss, train_op], feed_dict=feed_dict)

            duration = time.time() - start_time
            print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
            total_loss += loss_value

        save(saver, sess, FLAGS.log_dir, epoch)
        total_loss /= FLAGS.num_steps
        print('Epoch {:d} completed! Total Loss = {:.3f}'.format(epoch, total_loss))

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    main()
