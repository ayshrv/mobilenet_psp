from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import datetime
from PIL import Image
import numpy as np
np.set_printoptions(threshold=np.nan)
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

#Path to SemanticSegmentation Folder
SemanticSegmentation_DIR = '/home/n1703300e/SS/'

sys.path.insert(0, SemanticSegmentation_DIR+'Datasets/Cityscapes/cityscapesScripts/cityscapesscripts/helpers')
from labels import id2label, trainId2label

#Set Visible CUDA Devices
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
import tensorflow.contrib.slim as slim

PRINT_ARCHITECTURE = True
PRINT_INFO = True
DO_VALIDATION = True
OUTPUT_TO_FILE = False
USE_LATEST_WEIGHTS = False


#Dataset Flags
DATASET_DIR = SemanticSegmentation_DIR+'Datasets/Cityscapes/'
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 2048
NUM_READERS = 8
NUM_PREPROCESSING_THREADS = 8

#Model Flags
BATCH_SIZE = 1  #If changed, change process_mask_for_iou_loss also, as batch_size 1 assumed there.
NUM_CLASSES = 19
TRAIN_IMAGE_SIZE = 713

#Training Flags
START_EPOCH = 1
END_EPOCH = 200
NUM_EPOCHS = END_EPOCH - START_EPOCH + 1
OPTIMIZER = 'sgd'       #'rmsprop' or 'sgd'

START_LEARNING_RATE = 0.01
LEARNING_RATE_DECAY_FACTOR = 0.5
NUM_EPOCHS_PER_DELAY = 10.0

#Other Flags
LOGDIR = 'logs/GPU/test0_split_COARSE-FULL-SGD/'
OUTPUT_FILE= LOGDIR + 'log.txt'
PRETRAINED_CHECK_POINT = SemanticSegmentation_DIR+'MobileNet_self/MobileNetPreTrained/model.ckpt-906808'
MY_PRETRAINED_WEIGHTS = LOGDIR

#Change these files to change the Dataset, 'train_fine.tfRecord' for Fine Dataset / 'train_coarse.tfRecord' for Coarse Dataset and same for val.
dataset_filenames = {
    'train': 'train_fine.tfRecord',
    'val': 'val_fine.tfRecord'
}

#Number of images in dataset
NUM_SAMPLES = {
        'train': 2975,
        'val': 500,
}

_ITEMS_TO_DESCRIPTIONS = {
            'image_height' : 'height',
            'image_width' : 'width',
            'image_filename' : 'filename',
            'label': 'label',
            'image': 'A color image of varying height and width',
}

summaries_every_iter = []
summaries_costly = []
summaries_images = []
summaries_images_val=[]
TrainStep = tf.placeholder(tf.bool)
train_mean_loss_summary = []
val_mean_loss_summary = []

if OUTPUT_TO_FILE:
    if not osp.isdir(LOGDIR):
        os.makedirs(LOGDIR)
    sys.stdout=open(OUTPUT_FILE,"w")

def print_info():
    print('Start Epoch: %d' % (START_EPOCH))
    print('End Epoch: %d' % (END_EPOCH))
    print('No. Of Epochs: %d' % (NUM_EPOCHS))
    print('Total Running Steps: %d\n' % (NUM_EPOCHS*NUM_SAMPLES['train']))
    print('Batch Size: %d' % (BATCH_SIZE))
    print('Train Image Size: %d' % (TRAIN_IMAGE_SIZE))
    print('No. Of Classes: %d\n' % (NUM_CLASSES))
    print('Optimizer: %s' % (OPTIMIZER))
    print('Start learning Rate: %f' % (START_LEARNING_RATE))
    print('End learning Rate: ')
    print('Learning Rate Decay Factor: %f' % (LEARNING_RATE_DECAY_FACTOR))
    print('Decay Learning Rate After %d Epoch\n' % (NUM_EPOCHS_PER_DELAY))


def get_dataset(dataset_dir, split_name):
    if split_name not in ['train','val']:
        raise ValueError('split name %s was not recognized.' % split_name)

    if split_name=='train':
        file_name = dataset_filenames['train']
    else:
        file_name = dataset_filenames['val']

    file_pattern = os.path.join(dataset_dir, file_name)

    reader = tf.TFRecordReader

    keys_to_features = {
            'image/height': tf.FixedLenFeature(
                    [], dtype=tf.int64, default_value=-1),
            'image/width': tf.FixedLenFeature(
                    [], dtype=tf.int64, default_value=-1),
            'image/colorspace': tf.FixedLenFeature(
                    (), tf.string, default_value=''),
            'image/channels': tf.FixedLenFeature(
                    [], dtype=tf.int64, default_value=-1),
            'image/format': tf.FixedLenFeature(
                    (), tf.string, default_value='png'),
            'image/encoded': tf.FixedLenFeature(
                    (), tf.string, default_value=''),
            'image/filename': tf.FixedLenFeature(
                    (), tf.string, default_value=''),
            'label/height': tf.FixedLenFeature(
                    [], dtype=tf.int64, default_value=-1),
            'label/width': tf.FixedLenFeature(
                    [], dtype=tf.int64, default_value=-1),
            'label/colorspace': tf.FixedLenFeature(
                    (), tf.string, default_value=''),
            'label/channels': tf.FixedLenFeature(
                    [], dtype=tf.int64, default_value=-1),
            'label/format': tf.FixedLenFeature(
                    (), tf.string, default_value='png'),
            'label/encoded': tf.FixedLenFeature(
                    (), tf.string, default_value=''),
    }

    items_to_handlers = {
            'image_height' : slim.tfexample_decoder.Tensor('image/height'),
            'image_width' : slim.tfexample_decoder.Tensor('image/width'),
            'image': slim.tfexample_decoder.Image('image/encoded','image/format',shape=[1024,2048,3]),
            'image_filename' : slim.tfexample_decoder.Tensor('image/filename'),
            'label_height' : slim.tfexample_decoder.Tensor('label/height'),
            'label_width' : slim.tfexample_decoder.Tensor('label/width'),
            'label': slim.tfexample_decoder.Image('label/encoded','label/format',shape=[1024,2048,1],channels=1),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
            keys_to_features, items_to_handlers)

    dataset = slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=reader,
            decoder=decoder,
            num_samples=NUM_SAMPLES[split_name],
            items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
            num_classes=NUM_CLASSES,
            )
    return dataset

def get_cmap():
    cmap = np.zeros((256,3), dtype=np.uint8)
    for i in trainId2label.keys():
        cmap[i] = trainId2label[i].color
    return cmap

def labelToColorImage(labels):
    cmap = get_cmap()
    shape = list(labels.shape[:3])
    shape.append(3)
    color_image = np.zeros(shape, dtype=np.uint8)
    for i in range(color_image.shape[0]):
        for j in range(color_image.shape[1]):
            for k in range(color_image.shape[2]):
                color_image[i][j][k][:] = cmap[int(labels[i][j][k][0])]
    return color_image

def mobilenet_arg_scope(weight_decay=0.0):

    with slim.arg_scope(
      [slim.convolution2d, slim.separable_convolution2d, slim.convolution2d_transpose],
      weights_initializer=slim.initializers.xavier_initializer(),
      biases_initializer=slim.init_ops.zeros_initializer(),
      weights_regularizer=slim.l2_regularizer(weight_decay)) as sc:
        return sc

def mobilenet(inputs, num_classes=19, is_training=True, width_multiplier=1, scope='MobileNet'):

    num_classes = 19
    image_size = 713
    weight_decay=0.0
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

        depthwise_conv = slim.separable_convolution2d(inputs, num_outputs=None, stride=stride, depth_multiplier=1, kernel_size=kernel, scope=sc+'/depthwise_conv')
        bn = slim.batch_norm(depthwise_conv, scope=sc+'/dw_batch_norm')

        pointwise_conv = slim.convolution2d(bn, num_pwc_filters, kernel_size=[1, 1], scope=sc+'/pointwise_conv')
        bn = slim.batch_norm(pointwise_conv, scope=sc+'/pw_batch_norm')
        return bn

    def _pointwisePSP(inputs, num_pwc_filters, width_multiplier, sc):
        num_pwc_filters = round(num_pwc_filters * width_multiplier)
        pointwise_conv = slim.convolution2d(inputs, num_pwc_filters, kernel_size=[1, 1], scope=sc+'/pointwise_conv')
        bn = slim.batch_norm(pointwise_conv, scope=sc+'/pw_batch_norm')
        return bn

    arg_scope = mobilenet_arg_scope(weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
        with tf.variable_scope(scope) as sc:
            end_points_collection = sc.name + '_end_points'
            with slim.arg_scope([slim.separable_convolution2d, slim.separable_convolution2d, slim.convolution2d_transpose], activation_fn=None, outputs_collections=[end_points_collection]):
                with slim.arg_scope([slim.batch_norm], is_training=is_training, activation_fn=tf.nn.relu):
                    net = slim.convolution2d(inputs, round(32 * width_multiplier), [3, 3], stride=2, padding='SAME', scope='conv_1')
                    net = slim.batch_norm(net, scope='conv_1/batch_norm')
                    if PRINT_ARCHITECTURE: print('after conv_1: ',net)
                    net = _depthwise_separable_conv(net, 64, width_multiplier, sc='conv_ds_2')
                    if PRINT_ARCHITECTURE: print('after conv_ds_2: ',net)
                    net = _depthwise_separable_conv(net, 128, width_multiplier, downsample=True, sc='conv_ds_3')
                    if PRINT_ARCHITECTURE: print('after conv_ds_3: ',net)
                    net = _depthwise_separable_conv(net, 128, width_multiplier, sc='conv_ds_4')
                    if PRINT_ARCHITECTURE: print('after conv_ds_4: ',net)
                    net = _depthwise_separable_conv(net, 256, width_multiplier, downsample=True, sc='conv_ds_5')
                    if PRINT_ARCHITECTURE: print('after conv_ds_5: ',net)
                    net = _depthwise_separable_conv(net, 256, width_multiplier, sc='conv_ds_6')
                    if PRINT_ARCHITECTURE: print('after conv_ds_6: ',net)
                    net = _depthwise_separable_conv(net, 512, width_multiplier, downsample=False, sc='conv_ds_7')
                    if PRINT_ARCHITECTURE: print('after conv_ds_7: ',net)
                    net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_8')
                    if PRINT_ARCHITECTURE: print('after conv_ds_8: ',net)
                    net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_9')
                    if PRINT_ARCHITECTURE: print('after conv_ds_9: ',net)
                    net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_10')
                    if PRINT_ARCHITECTURE: print('after conv_ds_10: ',net)
                    net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_11')
                    if PRINT_ARCHITECTURE: print('after conv_ds_11: ',net)
                    net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_12')
                    if PRINT_ARCHITECTURE: print('after conv_ds_12: ',net)
                    # net = _depthwise_separable_conv(net, 1024, width_multiplier, downsample=False, sc='conv_ds_13')
                    # if PRINT_ARCHITECTURE: print('after conv_ds_13: ',net)
                    # net = _depthwise_separable_conv(net, 1024, width_multiplier, sc='conv_ds_14')
                    # if PRINT_ARCHITECTURE: print('after conv_ds_14: ',net)

                    net_a = slim.avg_pool2d(net, [90,90], stride=90, scope='conv_ds_15a/pool_15_1a')
                    if PRINT_ARCHITECTURE: print('after pool_15_1a: ',net_a)
                    net_b = slim.avg_pool2d(net, [45,45], stride=45, scope='conv_ds_15b/pool_15_1b')
                    if PRINT_ARCHITECTURE: print('after pool_15_1b: ',net_b)
                    net_c = slim.avg_pool2d(net, [30,30], stride=30, scope='conv_ds_15c/pool_15_1c')
                    if PRINT_ARCHITECTURE: print('after pool_15_1c: ',net_c)
                    net_d = slim.avg_pool2d(net, [15,15], stride=15, scope='conv_ds_15d/pool_15_1d')
                    if PRINT_ARCHITECTURE: print('after pool_15_1d: ',net_d)

                    net_a = _pointwisePSP(net_a, 128, width_multiplier, sc='conv_ds_15a/conv_ds_15_2a')
                    if PRINT_ARCHITECTURE: print('after conv_ds_15_2a: ',net_a)
                    net_b= _pointwisePSP(net_b, 128, width_multiplier, sc='conv_ds_15b/conv_ds_15_2b')
                    if PRINT_ARCHITECTURE: print('after conv_ds_15_2b: ',net_b)
                    net_c = _pointwisePSP(net_c, 128, width_multiplier, sc='conv_ds_15c/conv_ds_15_2c')
                    if PRINT_ARCHITECTURE: print('after conv_ds_15_2c: ',net_c)
                    net_d = _pointwisePSP(net_d, 128, width_multiplier, sc='conv_ds_15d/conv_ds_15_2d')
                    if PRINT_ARCHITECTURE: print('after conv_ds_15_2d: ',net_d)

                    # net_a = slim.convolution2d_transpose(net_a, 128, [90,90], [90,90],scope='conv_ds_15a/conv_t1')
                    # if PRINT_ARCHITECTURE: print('after conv_ds_15a/convt1: ',net_a)
                    # net_b = slim.convolution2d_transpose(net_b, 128, [45,45], [45,45],scope='conv_ds_15b/conv_t1')
                    # if PRINT_ARCHITECTURE: print('after conv_ds_15b/convt1: ',net_b)
                    # net_c = slim.convolution2d_transpose(net_c, 128, [30,30], [30,30],scope='conv_ds_15c/conv_t1')
                    # if PRINT_ARCHITECTURE: print('after conv_ds_15c/convt1: ',net_c)
                    # net_d = slim.convolution2d_transpose(net_d, 128, [15,15], [15,15],scope='conv_ds_15d/conv_t1')
                    # if PRINT_ARCHITECTURE: print('after conv_ds_15d/convt1: ',net_d)

                    net_a = tf.image.resize_bilinear(net_a, [90,90], align_corners=True, name='conv_ds_15a/conv_t1')
                    net_b = tf.image.resize_bilinear(net_b, [90,90], align_corners=True, name='conv_ds_15a/conv_t1')
                    net_c = tf.image.resize_bilinear(net_c, [90,90], align_corners=True, name='conv_ds_15a/conv_t1')
                    net_d = tf.image.resize_bilinear(net_d, [90,90], align_corners=True, name='conv_ds_15a/conv_t1')



                    fuse_15 = tf.concat([net, net_a,net_b,net_c,net_d],axis=3, name='fuse_psp')
                    if PRINT_ARCHITECTURE: print('after fuse_15: ',fuse_15)

                    net = _depthwise_separable_convPSP(fuse_15, [3,3], 1, 128, width_multiplier, sc='conv_ds_16')
                    if PRINT_ARCHITECTURE: print('after conv_ds_16: ',net)
                    net = _depthwise_separable_convPSP(net, [3,3], 1, 19, width_multiplier, sc='conv_ds_17')
                    if PRINT_ARCHITECTURE: print('after conv_ds_17: ',net)

                    net = slim.convolution2d_transpose(net, 19, [90,90], [7,7], padding='VALID', scope='conv_t2')
                    if PRINT_ARCHITECTURE: print('after conv_t2: ',net)

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            annotation_pred = tf.argmax(net, dimension=3, name="prediction")
    return tf.cast(tf.expand_dims(annotation_pred, dim=3),dtype=tf.uint8), end_points, net

def process_images(images):
    #[123.68, 116.78, 103.94]
    mean_img = tf.constant([123.68, 116.78, 103.94], dtype=tf.float32)
    mean_img = tf.reshape(mean_img, [1, 1, 3])
    images = tf.cast(images,dtype=tf.float32)
    images = images - mean_img
    images = tf.image.resize_images(images, [TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return images

def process_labels(labels):
    labels = tf.image.resize_images(labels, [TRAIN_IMAGE_SIZE,TRAIN_IMAGE_SIZE],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return tf.cast(labels,dtype=tf.float32)

def process_labels_for_cross_entropy_loss(labels):
    labels = tf.cast(tf.squeeze(labels,axis=3),dtype=tf.int32)
    invalid_labels_mask = tf.equal( labels, tf.constant( 255, dtype=tf.int32 ) )
    new_labels1 = tf.constant(-1, shape=labels.get_shape(),dtype = tf.int32 )
    new_labels = tf.Variable(new_labels1, name='processed_labels',trainable=False)
    new_labels = new_labels.assign(tf.where (invalid_labels_mask, new_labels, labels) )
    new_labels = tf.one_hot(new_labels,NUM_CLASSES)
    return new_labels

def process_mask_for_iou_loss(labels, predictions):
    labels = tf.cast(tf.squeeze(labels),dtype=tf.int32)
    invalid_labels_mask = tf.not_equal( labels, tf.constant( 255, dtype=tf.int32 ) )
    predictions = tf.cast(tf.squeeze(predictions), dtype=tf.int32)
    return labels, predictions, invalid_labels_mask

def forward_pass(train_batch_queue, val_batch_queue):
    images, labels, filename = tf.cond(TrainStep,lambda: train_batch_queue.dequeue(), lambda: val_batch_queue.dequeue())
    print("images: ",images)
    predictions, dict_feature_maps, final_map = mobilenet(images)
    print("labels: ",labels)
    print('\n')

    with tf.device('/cpu:0'):
        processed_labels_ce = process_labels_for_cross_entropy_loss(labels)
        processed_labels_iou, predictions_iou, mask = process_mask_for_iou_loss(labels, predictions)

    cross_entropies = tf.nn.softmax_cross_entropy_with_logits(labels=processed_labels_ce, logits=final_map)
    cross_entropy_mean = tf.reduce_mean(cross_entropies)


    with tf.device('/cpu:0'):
        print('processed_labels_for_accuracy: ',processed_labels_iou)
        print('predictions_for_accuracy: ',predictions_iou)
        print('mask_for_invalid_entries: ',mask)
        print('\n')
        # cf = tf.confusion_matrix(a,b,NUM_CLASSES, weights = c)
        # cf = tf.Print(cf,[cf.shape,cf],message='This is cf', summarize=19*19)
        iou, update_confusion_matrix1 = tf.metrics.mean_iou(processed_labels_iou, predictions_iou, NUM_CLASSES, weights = mask)
        mean_acc, update_confusion_matrix2 = tf.metrics.mean_per_class_accuracy(processed_labels_iou, predictions_iou, NUM_CLASSES, weights = mask)
        acc, update_confusion_matrix3 = tf.metrics.accuracy(processed_labels_iou, predictions_iou, weights = mask)
        update_confusion_matrix = [update_confusion_matrix1, update_confusion_matrix2, update_confusion_matrix3]

    with tf.device('/cpu:0'):
        color_labels = tf.py_func(labelToColorImage, [labels], [tf.uint8], name='labelsToColor')
        color_predictions = tf.py_func(labelToColorImage, [predictions], [tf.uint8], name='predictionsToColor')

        summaries_images_val.append(tf.summary.image('Validation/label',color_labels[0], max_outputs=BATCH_SIZE))
        summaries_images_val.append(tf.summary.image('Validation/Prediction',color_predictions[0], max_outputs=BATCH_SIZE))

        summaries_images.append(tf.summary.image('Training/0Image',images, max_outputs=BATCH_SIZE))
        summaries_images.append(tf.summary.image('Training/1LabelColor',color_labels[0], max_outputs=BATCH_SIZE))
        summaries_images.append(tf.summary.image('Training/1PredictionColor',color_predictions[0], max_outputs=BATCH_SIZE))
        summaries_images.append(tf.summary.image('Training/2LabelGrayscale',labels, max_outputs=BATCH_SIZE))
        summaries_images.append(tf.summary.image('Training/2PredictionsGrayscale',predictions, max_outputs=BATCH_SIZE))

        summaries_every_iter.append(tf.summary.scalar('Loss/CrossEntropyMean',cross_entropy_mean))
        summaries_every_iter.append(tf.summary.scalar('Accuracy/IoU',iou))
        summaries_every_iter.append(tf.summary.scalar('Accuracy/MeanAccuracy',mean_acc))
        summaries_every_iter.append(tf.summary.scalar('Accuracy/Accuracy',acc))

    return cross_entropy_mean, iou, mean_acc, acc, update_confusion_matrix

def get_batch_queue(split_name):
    dataset = get_dataset(DATASET_DIR, split_name)
    provider = slim.dataset_data_provider.DatasetDataProvider(
              dataset,
              num_readers=NUM_READERS,
              common_queue_capacity=20 * BATCH_SIZE,
              common_queue_min=10 * BATCH_SIZE)
    [image, label, image_filename] = provider.get(['image', 'label', 'image_filename'])
    print(str(split_name)+'_image: ',image)
    print(str(split_name)+'_label: ',label)
    resized_image = process_images(image)
    resized_label = process_labels(label)
    images, labels, image_filename = tf.train.batch(
              [resized_image, resized_label, image_filename],
              batch_size=BATCH_SIZE,
              num_threads=NUM_PREPROCESSING_THREADS)
    print(str(split_name)+'_resized_image: ',resized_image)
    print(str(split_name)+'_resized_label: ',resized_label)
    print('\n')
    batch_queue = slim.prefetch_queue.prefetch_queue(
              [images, labels, image_filename], capacity=2)
    return batch_queue

def weights_initialisers():
    if USE_LATEST_WEIGHTS:
        restoreAllVars = slim.get_variables_to_restore()
        print(
            'Ignoring --PRETRAINED_CHECK_POINT because a checkpoint already exists in %s'
             % MY_PRETRAINED_WEIGHTS)
        checkpoint_path = MY_PRETRAINED_WEIGHTS
        if tf.gfile.IsDirectory(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        return slim.assign_from_checkpoint_fn(checkpoint_path,restoreAllVars)
    restoreVar_mobilenet = slim.get_variables_to_restore(include=['MobileNet'],exclude=['MobileNet/conv_ds_15a','MobileNet/conv_ds_15b','MobileNet/conv_ds_15c','MobileNet/conv_ds_15d','MobileNet/conv_ds_16','MobileNet/conv_ds_17','MobileNet/conv_t2'])
    newLayerVariables = slim.get_variables_to_restore(include=['MobileNet/conv_ds_15a','MobileNet/conv_ds_15b','MobileNet/conv_ds_15c','MobileNet/conv_ds_15d','MobileNet/conv_ds_16','MobileNet/conv_ds_17','MobileNet/conv_t2'])
    optimizer_variables = slim.get_variables_to_restore(exclude=['MobileNet'])

    checkpoint_path=PRETRAINED_CHECK_POINT

    print('Checkpoint path: ',checkpoint_path)
    readMobileNetWeights = slim.assign_from_checkpoint_fn(checkpoint_path,restoreVar_mobilenet)
    otherLayerInitializer = tf.variables_initializer(newLayerVariables)
    restInitializer = tf.variables_initializer(optimizer_variables)
    return readMobileNetWeights, otherLayerInitializer, restInitializer


if PRINT_INFO:
            print_info()
with tf.device('/cpu:0'):

    train_batch_queue = get_batch_queue('train')
    val_batch_queue = get_batch_queue('val')

    train_mean_cross_entropy_placeholder = tf.placeholder(dtype=tf.float32, name='train_mean_cross_entropy')
    train_mean_iou_placeholder = tf.placeholder(dtype=tf.float32, name='train_mean_iou')
    train_mean_mean_acc_placeholder = tf.placeholder(dtype=tf.float32, name='train_mean_mean_acc')
    train_mean_acc_placeholder = tf.placeholder(dtype=tf.float32, name='train_mean_acc')

    train_mean_loss_summary.append(tf.summary.scalar('Epoch/TrainMeanCrossEntropy',train_mean_cross_entropy_placeholder))
    train_mean_loss_summary.append(tf.summary.scalar('Epoch/TrainMeanIoU',train_mean_iou_placeholder))
    train_mean_loss_summary.append(tf.summary.scalar('Epoch/TrainMeanPixelMeanAccuracy',train_mean_mean_acc_placeholder))
    train_mean_loss_summary.append(tf.summary.scalar('Epoch/TrainMeanAccuracy',train_mean_acc_placeholder))

    val_mean_cross_entropy_placeholder = tf.placeholder(dtype=tf.float32, name='val_mean_cross_entropy')
    val_mean_iou_placeholder = tf.placeholder(dtype=tf.float32, name='val_mean_iou')
    val_mean_mean_acc_placeholder = tf.placeholder(dtype=tf.float32, name='val_mean_mean_acc')
    val_mean_acc_placeholder = tf.placeholder(dtype=tf.float32, name='val_mean_acc')

    val_mean_loss_summary.append(tf.summary.scalar('Epoch/ValMeanCrossEntropy',val_mean_cross_entropy_placeholder))
    val_mean_loss_summary.append(tf.summary.scalar('Epoch/ValMeanIoU',val_mean_iou_placeholder))
    val_mean_loss_summary.append(tf.summary.scalar('Epoch/ValMeanPixelMeanAccuracy',val_mean_mean_acc_placeholder))
    val_mean_loss_summary.append(tf.summary.scalar('Epoch/ValMeanAccuracy',val_mean_acc_placeholder))

with tf.device('/gpu:0'):

    cross_entropy_mean, iou, mean_acc, acc, update_confusion_matrix = forward_pass(train_batch_queue,val_batch_queue)

with tf.device('/cpu:0'):

    rmsprop_decay =  0.9
    rmsprop_momentum = 0.9
    opt_epsilon = 1.0

    global_step = tf.Variable(0, trainable=False,name='global_step')
    decay_steps = int(NUM_SAMPLES['train'] / BATCH_SIZE * NUM_EPOCHS_PER_DELAY)
    learning_rate = tf.train.exponential_decay(START_LEARNING_RATE,
                                          global_step,
                                          decay_steps,
                                          LEARNING_RATE_DECAY_FACTOR,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    summaries_every_iter.append(tf.summary.scalar('TrainingParameters/learning_rate', learning_rate))
    summaries_every_iter.append(tf.summary.scalar('TrainingParameters/global_step', global_step))


with tf.device('/gpu:0'):
    with tf.variable_scope(OPTIMIZER):
        if OPTIMIZER =='rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate,decay=rmsprop_decay,momentum=rmsprop_momentum,epsilon=opt_epsilon)
        elif OPTIMIZER == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(loss=cross_entropy_mean)
        for grad_var_pair in gradients:
            current_variable = grad_var_pair[1]
            current_gradient = grad_var_pair[0]
            gradient_name_to_save = current_variable.name.replace(":", "_")
            with tf.device('/cpu:0'):
                summaries_costly.append(tf.summary.histogram(gradient_name_to_save, current_gradient))
        train_step = optimizer.apply_gradients(grads_and_vars=gradients,global_step=global_step)
        # print(train_step)

# restoreVar_mobilenet = slim.get_variables_to_restore(include=['MobileNet'],exclude=['MobileNet/conv_t1','MobileNet/conv_t2','MobileNet/conv_t3'])
# newLayerVariables = slim.get_variables_to_restore(include=['MobileNet/conv_t1','MobileNet/conv_t2','MobileNet/conv_t3'])
# optimizer_variables = slim.get_variables_to_restore(exclude=['MobileNet'])

with tf.device('/cpu:0'):

    if USE_LATEST_WEIGHTS:
        MobileNetAllWeightsFunction = weights_initialisers()
    else:
        MobileNetWeightsFunction, otherLayersInitializer, restInitializer =  weights_initialisers()
    localvariables = tf.initialize_local_variables()
    # merged_summary_op = tf.summary.merge_all()
    do_summaries_every_iter = tf.summary.merge(summaries_every_iter,name='every_iter_summaries')
    do_summaries_costly = tf.summary.merge(summaries_costly,name='costly_summaries')
    do_summaries_images = tf.summary.merge(summaries_images,name='images_summaries')
    do_summaries_images_val = tf.summary.merge(summaries_images_val,name='images_summaries_val')
    do_summaries_train = tf.summary.merge(train_mean_loss_summary,name='train_summaries')
    do_summaries_val = tf.summary.merge(val_mean_loss_summary,name='val_summaries')

    saver = tf.train.Saver(max_to_keep=0)

feed_dict_to_use={}
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
with tf.Session() as sess:

    # with tf.device('/cpu:0'):
        train_summary_writer = tf.summary.FileWriter(LOGDIR+'train/',graph=sess.graph)
        val_summary_writer = tf.summary.FileWriter(LOGDIR+'val/')
        if USE_LATEST_WEIGHTS:
            MobileNetAllWeightsFunction(sess)
        else:
            MobileNetWeightsFunction(sess)
            sess.run(otherLayersInitializer)
            sess.run(restInitializer)
        sess.run(localvariables)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            for itr in range(START_EPOCH,END_EPOCH+1):
                total_cross_entropy = 0.0
                total_iou = 0.0
                total_mean_acc = 0.0
                total_acc = 0.0
                ##Epoch Start
                print('\n\n**********************************************')
                print('Epoch: %d Start...' % (itr))
                print('**********************************************\n\n')
                for index in range(1,NUM_SAMPLES['train']+1):
                    #Train Step
                    global_step_index = (itr-1)*NUM_SAMPLES['train'] + index
                    feed_dict_to_use[TrainStep]=True
                    sess.run([train_step,update_confusion_matrix],feed_dict=feed_dict_to_use)
                    cross_entropy1, iou1, mean_acc1, acc1, summary_string= sess.run([cross_entropy_mean, iou, mean_acc, acc, do_summaries_every_iter],feed_dict=feed_dict_to_use)
                    train_summary_writer.add_summary(summary_string, global_step_index)
                    print("TimeStamp: %s  Epoch: %d Image: %04d/%d Cross Entropy Loss: %.5g\t Mean IoU Accuracy: %.5g\t Mean Pixel Accuracy: %.5g\t Pixel Accuracy: %.5g" % (datetime.datetime.now(),itr,index,NUM_SAMPLES['train'], cross_entropy1, iou1, mean_acc1, acc1))
                    total_cross_entropy += cross_entropy1
                    total_iou += iou1
                    total_mean_acc += mean_acc1
                    total_acc += acc1

                    if global_step_index%100==0:
                        summary_string = sess.run(do_summaries_costly,feed_dict=feed_dict_to_use)
                        train_summary_writer.add_summary(summary_string, global_step_index)

                    if global_step_index%500==0:
                        summary_string = sess.run(do_summaries_images,feed_dict=feed_dict_to_use)
                        train_summary_writer.add_summary(summary_string, global_step_index)
                ##Epoch End

                train_mean_cross_entropy = total_cross_entropy/NUM_SAMPLES['train']
                train_mean_iou = total_iou/NUM_SAMPLES['train']
                train_mean_mean_acc = total_mean_acc/NUM_SAMPLES['train']
                train_mean_acc = total_acc/NUM_SAMPLES['train']
                feed_dict_to_use[train_mean_cross_entropy_placeholder] = train_mean_cross_entropy
                feed_dict_to_use[train_mean_iou_placeholder] = train_mean_iou
                feed_dict_to_use[train_mean_mean_acc_placeholder] = train_mean_mean_acc
                feed_dict_to_use[train_mean_acc_placeholder] = train_mean_acc

                summary_string = sess.run(do_summaries_train,feed_dict=feed_dict_to_use)
                train_summary_writer.add_summary(summary_string, itr)


                print('\n\n**********************************************')
                print('Epoch: %d Completed... \nCross Entropy Loss: %.5g\t Mean IoU Accuracy: %.5g\t Mean Pixel Accuracy: %.5g\t Pixel Accuracy: %.5g' % (itr,train_mean_cross_entropy,train_mean_iou,train_mean_mean_acc, train_mean_acc))
                print('**********************************************\n\n')

                saver.save(sess, LOGDIR + "model_coarse.ckpt", itr)

                ##Validation Step
                if not DO_VALIDATION:
                    continue
                if itr % 1 != 0:
                    continue
                val_summary_img_writer = tf.summary.FileWriter(LOGDIR+'val_images_epoch-'+"%02d" %(itr)+'/')
                total_cross_entropy = 0.0
                total_iou = 0.0
                total_mean_acc = 0.0
                total_acc = 0.0
                print('Validation Started...')
                feed_dict_to_use[TrainStep]=False
                for index in range(1,NUM_SAMPLES['val']+1):
                    cross_entropy1, iou1, mean_acc1, acc1, summary_string= sess.run([cross_entropy_mean, iou, mean_acc, acc, do_summaries_images_val],feed_dict=feed_dict_to_use)
                    if index%50 == 0:
                        val_summary_img_writer.add_summary(summary_string, index)
                    print("TimeStamp: %s Validation Image: %03d/%d Cross Entropy Loss: %.5g\t Mean IoU Accuracy: %.5g\t Mean Pixel Accuracy: %.5g\t Pixel Accuracy: %.5g" % (datetime.datetime.now(), index, NUM_SAMPLES['val'], cross_entropy1, iou1, mean_acc1, acc1))
                    total_cross_entropy += cross_entropy1
                    total_iou += iou1
                    total_mean_acc += mean_acc1
                    total_acc += acc1
                val_mean_cross_entropy = total_cross_entropy/NUM_SAMPLES['val']
                val_mean_iou = total_iou/NUM_SAMPLES['val']
                val_mean_mean_acc = total_mean_acc/NUM_SAMPLES['val']
                val_mean_acc = total_acc/NUM_SAMPLES['val']
                print('\nValidation %d Complete....\nCross Entropy Loss: %.5g\t Mean IoU Accuracy: %.5g\t Mean Pixel Accuracy: %.5g\t Pixel Accuracy: %.5g' % (itr,val_mean_cross_entropy, val_mean_iou, val_mean_mean_acc, val_mean_acc))
                feed_dict_to_use[val_mean_cross_entropy_placeholder] = val_mean_cross_entropy
                feed_dict_to_use[val_mean_iou_placeholder] = val_mean_iou
                feed_dict_to_use[val_mean_mean_acc_placeholder] = val_mean_mean_acc
                feed_dict_to_use[val_mean_acc_placeholder] = val_mean_acc

                summary_string = sess.run(do_summaries_val,feed_dict=feed_dict_to_use)
                val_summary_writer.add_summary(summary_string, itr)

        finally:
            coord.request_stop()
            coord.join(threads)

sys.stdout.close()
