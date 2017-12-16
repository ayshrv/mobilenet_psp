import tensorflow as tf
slim = tf.contrib.slim

def MobileNet(inputs, num_classes=19, is_training=True, width_multiplier=1, scope='MobileNet', print_architecture=False):

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
                    if print_architecture: print('after conv_1: ',net)
                    net = _depthwise_separable_conv(net, 64, width_multiplier, sc='conv_ds_2')
                    if print_architecture: print('after conv_ds_2: ',net)
                    net = _depthwise_separable_conv(net, 128, width_multiplier, downsample=True, sc='conv_ds_3')
                    if print_architecture: print('after conv_ds_3: ',net)
                    net = _depthwise_separable_conv(net, 128, width_multiplier, sc='conv_ds_4')
                    if print_architecture: print('after conv_ds_4: ',net)
                    net = _depthwise_separable_conv(net, 256, width_multiplier, downsample=True, sc='conv_ds_5')
                    if print_architecture: print('after conv_ds_5: ',net)
                    net = _depthwise_separable_conv(net, 256, width_multiplier, sc='conv_ds_6')
                    if print_architecture: print('after conv_ds_6: ',net)
                    net = _depthwise_separable_conv(net, 512, width_multiplier, downsample=False, sc='conv_ds_7')
                    if print_architecture: print('after conv_ds_7: ',net)
                    net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_8')
                    if print_architecture: print('after conv_ds_8: ',net)
                    net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_9')
                    if print_architecture: print('after conv_ds_9: ',net)
                    net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_10')
                    if print_architecture: print('after conv_ds_10: ',net)
                    net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_11')
                    if print_architecture: print('after conv_ds_11: ',net)
                    net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_12')
                    if print_architecture: print('after conv_ds_12: ',net)
                    net = _depthwise_separable_conv(net, 1024, width_multiplier, downsample=False, sc='conv_ds_13')
                    if print_architecture: print('after conv_ds_13: ',net)
                    net = _depthwise_separable_conv(net, 1024, width_multiplier, sc='conv_ds_14')
                    if print_architecture: print('after conv_ds_14: ',net)

                    net_a = slim.avg_pool2d(net, [90,90], stride=90, scope='conv_ds_15a/pool_15_1a')
                    if print_architecture: print('after pool_15_1a: ',net_a)
                    net_b = slim.avg_pool2d(net, [45,45], stride=45, scope='conv_ds_15b/pool_15_1b')
                    if print_architecture: print('after pool_15_1b: ',net_b)
                    net_c = slim.avg_pool2d(net, [30,30], stride=30, scope='conv_ds_15c/pool_15_1c')
                    if print_architecture: print('after pool_15_1c: ',net_c)
                    net_d = slim.avg_pool2d(net, [15,15], stride=15, scope='conv_ds_15d/pool_15_1d')
                    if print_architecture: print('after pool_15_1d: ',net_d)

                    net_a = _pointwisePSP(net_a, 256, width_multiplier, sc='conv_ds_15a/conv_ds_15_2a')
                    if print_architecture: print('after conv_ds_15_2a: ',net_a)
                    net_b= _pointwisePSP(net_b,  256, width_multiplier, sc='conv_ds_15b/conv_ds_15_2b')
                    if print_architecture: print('after conv_ds_15_2b: ',net_b)
                    net_c = _pointwisePSP(net_c, 256, width_multiplier, sc='conv_ds_15c/conv_ds_15_2c')
                    if print_architecture: print('after conv_ds_15_2c: ',net_c)
                    net_d = _pointwisePSP(net_d, 256, width_multiplier, sc='conv_ds_15d/conv_ds_15_2d')
                    if print_architecture: print('after conv_ds_15_2d: ',net_d)

                    net_a = tf.image.resize_bilinear(net_a, [90,90], align_corners=True, name='conv_ds_15a/conv_t1')
                    net_b = tf.image.resize_bilinear(net_b, [90,90], align_corners=True, name='conv_ds_15b/conv_t1')
                    net_c = tf.image.resize_bilinear(net_c, [90,90], align_corners=True, name='conv_ds_15c/conv_t1')
                    net_d = tf.image.resize_bilinear(net_d, [90,90], align_corners=True, name='conv_ds_15d/conv_t1')

                    fuse_15 = tf.concat([net, net_a,net_b,net_c,net_d],axis=3, name='fuse_psp')
                    if print_architecture: print('after fuse_15: ',fuse_15)

                    net = _depthwise_separable_convPSP(fuse_15, [3,3], 1, 256, width_multiplier, sc='conv_ds_16')
                    if print_architecture: print('after conv_ds_16: ',net)
                    net = _depthwise_separable_convPSP(net, [3,3], 1, 19, width_multiplier, sc='conv_ds_17')
                    if print_architecture: print('after conv_ds_17: ',net)

            # annotation_pred = tf.argmax(net, dimension=3, name="prediction")
    return net
    # return tf.cast(tf.expand_dims(annotation_pred, dim=3),dtype=tf.uint8), net
