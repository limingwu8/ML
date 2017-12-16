import tensorflow as tf
import utils


# %%
def VGG16(x, n_classes, is_pretrain=True):
    x = utils.conv('conv1_1', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = utils.conv('conv1_2', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = utils.pool('pool1', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

    x = utils.conv('conv2_1', x, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = utils.conv('conv2_2', x, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = utils.pool('pool2', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

    x = utils.conv('conv3_1', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = utils.conv('conv3_2', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = utils.conv('conv3_3', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = utils.pool('pool3', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

    x = utils.conv('conv4_1', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = utils.conv('conv4_2', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = utils.conv('conv4_3', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = utils.pool('pool3', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

    x = utils.conv('conv5_1', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = utils.conv('conv5_2', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = utils.conv('conv5_3', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = utils.pool('pool3', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

    x = utils.FC_layer('fc6', x, out_nodes=4096)
    x = utils.batch_norm(x)
    x = utils.FC_layer('fc7', x, out_nodes=4096)
    x = utils.batch_norm(x)
    x = utils.FC_layer('fc8', x, out_nodes=n_classes)

    return x


# %% TO get better tensorboard figures!

def VGG16N(x, n_classes, is_pretrain=True):
    with tf.name_scope('VGG16'):
        x = utils.conv('conv1_1', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = utils.conv('conv1_2', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        with tf.name_scope('pool1'):
            x = utils.pool('pool1', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = utils.conv('conv2_1', x, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = utils.conv('conv2_2', x, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        with tf.name_scope('pool2'):
            x = utils.pool('pool2', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = utils.conv('conv3_1', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = utils.conv('conv3_2', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = utils.conv('conv3_3', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        with tf.name_scope('pool3'):
            x = utils.pool('pool3', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = utils.conv('conv4_1', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = utils.conv('conv4_2', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = utils.conv('conv4_3', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        with tf.name_scope('pool4'):
            x = utils.pool('pool4', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = utils.conv('conv5_1', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = utils.conv('conv5_2', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = utils.conv('conv5_3', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        with tf.name_scope('pool5'):
            x = utils.pool('pool5', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = utils.FC_layer('fc6', x, out_nodes=4096)
        with tf.name_scope('batch_norm1'):
            x = utils.batch_norm(x)
        x = utils.FC_layer('fc7', x, out_nodes=4096)
        with tf.name_scope('batch_norm2'):
            x = utils.batch_norm(x)
        x = utils.FC_layer('fc8', x, out_nodes=n_classes)

        return x