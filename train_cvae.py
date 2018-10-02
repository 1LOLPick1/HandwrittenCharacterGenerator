from cvae import create_cvae
import argparse
import logging
import pandas as pd
import random
import numpy as np
from collections import defaultdict
import cv2
from keras.callbacks import TensorBoard, ModelCheckpoint
import os
from generate_dataset_csv import generate_paths


classes = ['J', 'K', 'L', 'M', 'N', 'O', 'Z', 'j', 'k', 'l', 'm', 'n', 'o',
           'z', '0', '1', '2', '3',
               '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
           'G', 'H', 'I', 'P',
               'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'a', 'b', 'c',
           'd', 'e', 'f', 'g', 'h', 'i', 'p',
               'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']


def one_hot_vector(label):
    res = [0]*len(classes)
    res[classes.index(label)] = 1
    return np.array(res, dtype=np.float32)


def resize_coeff(x, new_x):
    """
    Evaluate resize coefficient from image shape
    Args:
        x: original value
        new_x: expect value

    Returns:
        Resize coefficient
    """
    return new_x / x


def resize_image(img, resize_shape=(128, 128), interpolation=cv2.INTER_AREA):
    """
    Resize single image
    Args:
        img: input image
        resize_shape: resize shape in format (height, width)
        interpolation: interpolation method

    Returns:
        Resized image
    """
    return cv2.resize(img, None, fx=resize_coeff(img.shape[1], resize_shape[1]),
                     fy=resize_coeff(img.shape[0], resize_shape[0]),
                     interpolation=interpolation)


def get_actual_area(img, threshold=200):
    x0, x1 = 0, 0
    y0, y1 = 0, 0

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] < 200:
                if y0 == 0:
                    y0 = i
                y1 = i

    for j in range(img.shape[1]):
        for i in range(img.shape[0]):
            if img[i, j] < threshold:
                if x0 == 0:
                    x0 = j
                x1 = j

    width = x1 - x0
    height = y1 - y0

    d = width - height

    if d < 0:
        x0 += d // 2
        x1 -= d // 2
    else:
        y0 -= d // 2
        y1 += d // 2

    width = x1 - x0
    height = y1 - y0

    x1 -= width - height

    return img[y0:y1 + 1, x0:x1 + 1]


def read_sample(sample, img_shape):
    img_path = sample['image_path']
    label = sample['label']

    img = cv2.imread(img_path, 0)

    if img is None:
        logging.info('Can\'t open image: ' + img_path)
        raise IOError('Image with path: ' + img_path + ', not found')

    img = resize_image(img, img_shape)

    return img / 255., one_hot_vector(label)


def dataset_generator(dataset, batch_size, img_shape, infinite=True):
    random.seed()

    while True:
        shufled_dataset = dataset.copy() #dataset.sample(frac=1.0)
        random.shuffle(shufled_dataset)
        imgs_batch, labels_batch = [], []
        for item in shufled_dataset:
            logging.debug('item:{}'.format(item))

            try:
                img, label = read_sample({'label':item[0], 'image_path': item[1]}, img_shape)
                pass
            except IOError:
                continue

            if img is None or label is None:
                continue

            imgs_batch.append(img)
            labels_batch.append(label)

            if len(imgs_batch) >= batch_size:
                yield [
                          np.reshape(
                              np.array(imgs_batch),
                              (batch_size, img_shape[0], img_shape[0], 1)
                          ),
                          np.array(labels_batch),
                          np.array(labels_batch)
                      ], np.reshape(
                              np.array(imgs_batch),
                              (batch_size, img_shape[0], img_shape[0], 1)
                          )
                imgs_batch, labels_batch = [], []

        if not infinite:
            break


def open_dataset(dataset_path, batch_size, img_shape, infinite=True):
    """Open dataset and return (generator, batches_count)."""
    dataset = generate_paths()

    dataset_gen = dataset_generator(
        dataset,
        batch_size=batch_size, infinite=infinite,
        img_shape=img_shape
    )
    steps = len(dataset) // batch_size
    return dataset_gen, steps


def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('train_dataset',
                            help='Path to csv file with train dataset.')
    arg_parser.add_argument('--validation-dataset',
                            help='Path to csv file with dataset info')
    arg_parser.add_argument('--batch-size', default=32, type=int,
                            help='Size of batch of images.')
    arg_parser.add_argument('--epochs', default=5, type=int,
                            help='Number of epochs.')
    arg_parser.add_argument('--logdir', help='Path to save tensorboard logs.')
    arg_parser.add_argument('--checkpoints', help='Path to save model weights.')
    arg_parser.add_argument('--loglevel', required=False, default='info',
                            choices=['info', 'debug', 'error'], type=str,
                            help=
                            'Choice logging level. Can be: info,debug,error.')
    arg_parser.add_argument('--image-width', required=False,
                            type=int,
                            default=28)
    arg_parser.add_argument('--latentdim', required=False,
                            type=int,
                            default=2)
    return arg_parser.parse_args()


def set_logger(loglevel='info'):
    if loglevel == 'info':
        logging.basicConfig(level=logging.INFO)

    if loglevel == 'debug':
        logging.basicConfig(level=logging.DEBUG)

    if loglevel == 'error':
        logging.basicConfig(level=logging.ERROR)


if __name__ == '__main__':
    app_args = parse_arguments()

    set_logger(app_args.loglevel)

    datasets = [('train', app_args.train_dataset),
                ('val', app_args.validation_dataset)]

    datasets = {
        ds_type: open_dataset(path, app_args.batch_size,
                              (app_args.image_width, app_args.image_width))
        for ds_type, path in datasets
        if path
    }

    datasets = defaultdict(lambda: (None, None), datasets)

    models, loss = create_cvae(
        input_shape=(app_args.image_width, app_args.image_width),
        num_classes=len(classes),
        latent_dim=app_args.latentdim,
        dropout_rate=0.4,
        batch_size=app_args.batch_size
    )

    cvae = models["cvae"]

    logging.info(cvae.summary())

    callbacks = []
    if app_args.logdir:
        callbacks.append(TensorBoard(log_dir=app_args.logdir, write_grads=True,
                                     write_images=True))
        if not app_args.checkpoints:
            app_args.checkpoints = app_args.logdir

    if app_args.checkpoints:
        callbacks.append(ModelCheckpoint(
            os.path.join(app_args.checkpoints, 'cvae-{epoch}.h5'.format(
                epoch='{epoch}')),
            save_weights_only=True
        ))

    cvae.fit_generator(
        datasets['train'][0],
        steps_per_epoch=datasets['train'][1],
        validation_data=datasets['val'][0],
        validation_steps=datasets['val'][1],
        callbacks=callbacks,
        epochs=app_args.epochs
    )



