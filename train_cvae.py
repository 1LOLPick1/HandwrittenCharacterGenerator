import argparse
import logging
import pandas as pd
import random
import numpy as np
from collections import defaultdict
import cv2
from keras.callbacks import TensorBoard, ModelCheckpoint
import os
import re

from generate_dataset_csv import generate_paths
from cvae import create_cvae


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
        shufled_dataset = dataset.copy()
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


def load_last_weights(model, path):
    if not os.path.isdir(path):
        os.makedirs(path)

    weights_files_list = [
        matching_f.group()
        for matching_f in map(
            lambda x: re.match('cvae-\d+-\d+-\d+-\d+.h5', x),
            os.listdir(path)
        ) if matching_f if not None
    ]

    if len(weights_files_list) == 0:
        return 0

    weights_files_list.sort(key=lambda x: -int(x.split('-')[1]))

    model.load_weights(os.path.join(path, weights_files_list[0]))

    logging.debug('LOAD MODEL PATH: {}'.format(
        os.path.join(path, weights_files_list[0])
    ))

    return int(weights_files_list[0].split('-')[1])


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

    start_epoch = load_last_weights(cvae, app_args.checkpoints)
    logging.info('START EPOCH NUMBER: {}'.format(start_epoch))

    logging.info(cvae.summary())

    callbacks = []
    if app_args.logdir:
        callbacks.append(TensorBoard(log_dir=app_args.logdir, write_grads=True,
                                     write_images=True))
        if not app_args.checkpoints:
            app_args.checkpoints = app_args.logdir

    if app_args.checkpoints:
        callbacks.append(ModelCheckpoint(
            os.path.join(
                app_args.checkpoints,
                'cvae-{epoch}-{latent}-{batch}-{img_width}.h5'.format(
                    epoch='{epoch}',
                    latent=app_args.latentdim,
                    batch=app_args.batch_size,
                    img_width=app_args.image_width
                )
            ),
            save_weights_only=True
        ))

    cvae.fit_generator(
        datasets['train'][0],
        steps_per_epoch=datasets['train'][1],
        validation_data=datasets['val'][0],
        validation_steps=datasets['val'][1],
        callbacks=callbacks,
        epochs=app_args.epochs,
        initial_epoch=start_epoch
    )



