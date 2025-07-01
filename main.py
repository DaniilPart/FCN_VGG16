import helper
import os
import warnings
import tensorflow as tf
import tests as tests
import scipy.misc

import progressbar
from termcolor import colored
from tensorflow.python.util import compat
from distutils.version import LooseVersion
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2

image_shape = (160, 576)

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name('image_input:0')
    keep = graph.get_tensor_by_name('keep_prob:0')
    layer3 = graph.get_tensor_by_name('layer3_out:0')
    layer4 = graph.get_tensor_by_name('layer4_out:0')
    layer7 = graph.get_tensor_by_name('layer7_out:0')
    return w1, keep, layer3, layer4, layer7


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network. Build skip-layers using the vgg layers.
    """
    l2_value = 1e-3
    kernel_reg = tf.contrib.layers.l2_regularizer(l2_value)
    stddev = 1e-3
    kernel_init = tf.random_normal_initializer(stddev=stddev)

    conv_1x1_7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                                  kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)

    conv7_2x = tf.layers.conv2d_transpose(conv_1x1_7, num_classes, 4, strides=2,
                                          padding='same', kernel_regularizer=kernel_reg,
                                          kernel_initializer=kernel_init)

    conv_1x1_4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
                                  kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)

    skip_4_to_7 = tf.add(conv7_2x, conv_1x1_4)

    upsample2x_skip_4_to_7 = tf.layers.conv2d_transpose(skip_4_to_7, num_classes, 4, strides=2,
                                                        padding='same', kernel_regularizer=kernel_reg,
                                                        kernel_initializer=kernel_init)

    conv_1x1_3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                                  kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)

    skip_3 = tf.add(upsample2x_skip_4_to_7, conv_1x1_3)

    output = tf.layers.conv2d_transpose(skip_3, num_classes, 16, strides=8,
                                        padding='same', kernel_regularizer=kernel_reg,
                                        kernel_initializer=kernel_init)
    return output


def get_logits(last_layer, num_classes):
    return tf.reshape(last_layer, (-1, num_classes))


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    logits = get_logits(nn_last_layer, num_classes)
    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 0.01
    loss = cross_entropy_loss + reg_constant * sum(reg_losses)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)
    return logits, train_op, loss


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss,
             input_image, correct_label, keep_prob, learning_rate):
    for epoch in range(epochs):
        print('##############################################################')
        print('..............Training Epoch # {}/{}..............'.format(epoch, epochs))
        print('##############################################################')
        bar = progressbar.ProgressBar()
        loss = None
        for image, label in bar(get_batches_fn(batch_size)):
            feed_dict = {
                input_image: image,
                correct_label: label,
                keep_prob: 0.5,
                learning_rate: learn_rate
            }
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)
        print('\nTraining Loss = {:.3f}'.format(loss))


def graph_visualize():
    with tf.Session() as sess:
        model_filename = os.path.join(vgg_dir, 'saved_model.pb')
        with gfile.FastGFile(model_filename, 'rb') as f:
            data = compat.as_bytes(f.read())
            sm = saved_model_pb2.SavedModel()
            sm.ParseFromString(data)
            tf.import_graph_def(sm.meta_graphs[0].graph_def)
    train_writer = tf.summary.FileWriter(log_dir)
    train_writer.add_graph(sess.graph)


def run():
    runs_dir = './runs'
    print("\n\nTesting for dataset presence.....")
    tests.test_looking_for_dataset(data_dir)

    helper.maybe_download_pretrained_vgg(vgg_dir)

    with tf.Session() as sess:
        get_batches_fn = helper.gen_batch_function(
            glob_trainig_images_path, glob_labels_trainig_image_path, image_shape)

        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_dir)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss,
                 input_image, correct_label, keep_prob, learning_rate)

        saver = tf.train.Saver()
        folderToSaveModel = "model"
        if not os.path.exists(folderToSaveModel):
            os.makedirs(folderToSaveModel)
        pathSaveModel = os.path.join(folderToSaveModel, "model.ckpt")
        saver.save(sess, pathSaveModel)
        print(colored("Model saved in path: {}".format(pathSaveModel), 'green'))

        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape,
                                      logits, keep_prob, input_image)


def all_is_ok():
    assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), \
        'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
    print('TensorFlow Version: {}'.format(tf.__version__))

    if not tf.test.gpu_device_name():
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

    print("\n\nTesting load_vgg function......")
    tests.test_load_vgg(load_vgg, tf)

    print("\n\nTesting layers function......")
    tests.test_layers(layers)

    print("\n\nTesting optimize function......")
    tests.test_optimize(optimize)

    print("\n\nTesting train_nn function......")
    tests.test_train_nn(train_nn)


def predict_by_model():
    if path_data is False and pred_data_from not in ['video', 'image']:
        exit("Path or mode not set correctly")

    vgg_path = os.path.join('./data', 'vgg')
    tf_config = tf.ConfigProto(device_count={'GPU': 0}) if disable_gpu else tf.ConfigProto()

    with tf.Session(config=tf_config) as sess:
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits = get_logits(nn_last_layer, num_classes)

        saver = tf.train.Saver()
        saver.restore(sess, path_model)

        if pred_data_from == 'video':
            helper.predict_video(path_data, sess, image_shape, logits, keep_prob, input_image)
        elif pred_data_from == 'image':
            image = scipy.misc.imresize(scipy.misc.imread(path_data), image_shape)
            street_im = helper.predict(sess, image, input_image, keep_prob, logits, image_shape)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            imagePath = os.path.join(current_dir, "image_predicted.png")
            scipy.misc.imsave(imagePath, street_im)
            print(colored("Image saved in {}".format(imagePath), 'green'))


if __name__ == '__main__':
    (disable_gpu,
     pred_data_from,
     path_model,
     path_data,
     num_classes,
     epochs,
     batch_size,
     vgg_dir,
     learn_rate,
     log_dir,
     data_dir,
     graph_visualize,
     glob_trainig_images_path,
     glob_labels_trainig_image_path) = helper.get_args()

    if not path_model:
        all_is_ok()
        run()
    else:
        predict_by_model()

    if graph_visualize:
        print("\n\nConverting .pb file to TF Summary and Saving Visualization of VGG16 graph..............")
        graph_visualize()
