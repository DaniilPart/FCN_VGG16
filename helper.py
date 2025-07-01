import re
import shutil
import random
import zipfile
import os.path
import scipy.misc
import numpy as np
import tensorflow as tf
import time
import subprocess

from glob import glob
from tqdm import tqdm
from optparse import OptionParser
from urllib.request import urlretrieve

from VideoGet import VideoGet
from VideoShow import VideoShow
# ZED kamera odstraněna – import VideoZed byl smazán


class DLProgress(tqdm):
    last_block = 0
    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(vgg_path):
    vgg_filename = 'vgg.zip'
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]
    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(os.path.join(vgg_path, '..'))
        zip_ref.close()
        os.remove(os.path.join(vgg_path, vgg_filename))


def get_args():
    parser = OptionParser()
    parser.add_option("-i", "--glob_trainig_images_path", dest="glob_trainig_images_path")
    parser.add_option("-l", "--glob_labels_trainig_image_path", dest="glob_labels_trainig_image_path")
    parser.add_option("-r", "--learn_rate", dest="learn_rate")
    parser.add_option("-n", "--num_classes", dest="num_classes")
    parser.add_option("-e", "--epochs", dest="epochs")
    parser.add_option("-b", "--batch_size", dest="batch_size")
    parser.add_option("-t", "--data_path", dest="data_path")
    parser.add_option("-p", "--log_path", dest="log_path")
    parser.add_option("-v", "--vgg_dir", dest="vgg_dir")
    parser.add_option("-g", "--graph_visualize", dest="graph_visualize")
    parser.add_option("-m", "--path_model", dest="path_model")
    parser.add_option("-V", "--path_data", dest="path_data")
    parser.add_option("", "--pred_data_from", dest="pred_data_from", help="Choose type [video, image]")
    parser.add_option("", "--disable_gpu", dest="disable_gpu", action="store_true")

    (options, args) = parser.parse_args()

    log_path = options.log_path if options.log_path is not None else '.'
    epochs = int(options.epochs) if options.epochs is not None else 25
    batch_size = options.batch_size if options.batch_size is not None else 4
    vgg_dir = options.vgg_dir if options.vgg_dir is not None else './data/vgg'
    learn_rate = float(options.learn_rate) if options.learn_rate is not None else 9e-5
    data_path = options.data_path if options.data_path is not None else './data/data_road'
    graph_visualize = options.graph_visualize if options.graph_visualize is not None else False
    num_classes = options.num_classes if options.num_classes is not None else 2
    glob_trainig_images_path = options.glob_trainig_images_path if options.glob_trainig_images_path \
        is None else './data/data_road/training/image_2/*.png'
    glob_labels_trainig_image_path = options.glob_labels_trainig_image_path if options.glob_labels_trainig_image_path \
        is None else './data/data_road/training/gt_image_2/*_road_*.png'
    path_model = options.path_model if options.path_model is not None else False
    path_data = options.path_data if options.path_data is not None else False
    pred_data_from = options.pred_data_from if options.pred_data_from is not None else "video"
    disable_gpu = True if options.disable_gpu is not None else False

    return (disable_gpu,
            pred_data_from,
            path_model,
            path_data,
            int(num_classes),
            epochs,
            batch_size,
            vgg_dir,
            learn_rate,
            log_path,
            data_path,
            graph_visualize,
            glob_trainig_images_path,
            glob_labels_trainig_image_path)


def gen_batch_function(glob_trainig_images_path, glob_labels_trainig_image_path, image_shape):
    def get_batches_fn(batch_size):
        image_paths = glob(glob_trainig_images_path)
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(glob_labels_trainig_image_path)}
        background_color = np.array([255, 0, 0])
        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images, gt_images = [], []
            for image_file in image_paths[batch_i:batch_i + batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]
                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
                gt_bg = np.all(gt_image == background_color, axis=2).reshape(*gt_image.shape[:2], 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
                images.append(image)
                gt_images.append(gt_image.astype(np.float32))
            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        start = time.clock()
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        street_im = predict(sess, image, image_pl, keep_prob, logits, image_shape)
        timeCount = (time.time() - start)
        print(image_file + " process time: " + str(timeCount))
        subprocess.call("echo {} - {} >> ./data/data_road/time.txt".format(image_file, timeCount), shell=True)
        yield os.path.basename(image_file), np.array(street_im)


def predict(sess, image, image_pl, keep_prob, logits, image_shape):
    im_softmax = sess.run([tf.nn.softmax(logits)], {keep_prob: 1.0, image_pl: [image]})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)
    return street_im


def predict_video(data_dir, sess, image_shape, logits, keep_prob, input_image):
    print('Predicting Video...')
    video_getter = VideoGet(data_dir, sess, image_shape, logits, keep_prob, input_image).start()
    video_shower = VideoShow(video_getter.frame).start()
    while True:
        if video_getter.stopped or video_shower.stopped:
            video_shower.stop()
            video_getter.stop()
            break
        frame = video_getter.frame
        video_shower.frame = frame
    # Funkce read_zed byla odstraněna


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    output_dir = os.path.join(runs_dir, str(int(time.time())))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(sess, logits, keep_prob, input_image,
                                    os.path.join(data_dir, 'testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
