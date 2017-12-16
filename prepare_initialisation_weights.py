from __future__ import print_function

import numpy as np
import tensorflow as tf

import os

from mobilenet import MobileNet

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'pretrained_mobilenet', 'MobileNetPreTrained/model.ckpt-906808',
    'Directory where pretrained mobilenet weights are located.')

tf.app.flags.DEFINE_string(
    'save_model', 'MobileNetPSP',
    'Directory where weights are to be saved')

tf.app.flags.DEFINE_boolean('print_architecture', True,
                            'Print architecure.')

tf.app.flags.DEFINE_integer('gpu', 0,
                            'Which GPU to use.')

FLAGS = tf.app.flags.FLAGS

#Set Visible CUDA Devices
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(FLAGS.gpu)

def weights_initialisers():
    new_variables = ['MobileNet/conv_ds_15a',
                     'MobileNet/conv_ds_15b',
                     'MobileNet/conv_ds_15c',
                     'MobileNet/conv_ds_15d',
                     'MobileNet/conv_ds_16',
                     'MobileNet/conv_ds_17']
    restoreVar_mobilenet = slim.get_variables_to_restore(include=['MobileNet'],exclude=new_variables)
    # restoreVar_mobilenet = [v for v in restoreVar_mobilenet if 'Momentum' not in v.name]
    newLayerVariables = slim.get_variables_to_restore(include=new_variables)

    checkpoint_path=FLAGS.pretrained_check_point
    print('Restoring weights from  %s: ' % (checkpoint_path) )
    readMobileNetWeights = slim.assign_from_checkpoint_fn(checkpoint_path,restoreVar_mobilenet)
    otherLayerInitializer = tf.variables_initializer(newLayerVariables)
    return readMobileNetWeights, otherLayerInitializer

def save(saver, sess, logdir):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, write_meta_graph=False)
    print('The weights have been saved to {}.'.format(checkpoint_path))

def main():

    tf.set_random_seed(1234)

    image_batch = tf.constant(0, tf.float32, shape=[1, 713, 713, 3])
    net = MobileNet(image_batch, print_architecture=True)

    new_variables = ['MobileNet/conv_ds_15a',
                     'MobileNet/conv_ds_15b',
                     'MobileNet/conv_ds_15c',
                     'MobileNet/conv_ds_15d',
                     'MobileNet/conv_ds_16',
                     'MobileNet/conv_ds_17']
    restoreVar_mobilenet = slim.get_variables_to_restore(include=['MobileNet'],exclude=new_variables)
    # restoreVar_mobilenet = [v for v in restoreVar_mobilenet if 'Momentum' not in v.name]
    newLayerVariables = slim.get_variables_to_restore(include=new_variables)
    otherLayerInitializer = tf.variables_initializer(newLayerVariables)

    var_list = tf.global_variables()

    # init = tf.global_variables_initializer()

    # loader = tf.train.Saver(var_list=restoreVar_mobilenet)
    # loader.restore(sess, FLAGS.pretrained_check_point)


    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # init = tf.global_variables_initializer()
        # sess.run(init)

        loader = tf.train.Saver(var_list=restoreVar_mobilenet)
        loader.restore(sess, FLAGS.pretrained_mobilenet)

        sess.run(otherLayerInitializer)

        # Saver for converting the loaded weights into .ckpt.
        saver = tf.train.Saver(var_list=var_list)
        save(saver, sess, FLAGS.save_model)

if __name__ == '__main__':
    main()
