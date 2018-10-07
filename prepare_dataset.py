import cv2
import numpy as np
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from generate_dataset_csv import generate_paths


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


def imap_unordered_bar(func, args, n_processes=8):
    p = Pool(n_processes)
    res_list = []
    with tqdm(total = len(args)) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(func, args))):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list


def argument_parser():
    arg_pars = argparse.ArgumentParser()
    arg_pars.add_argument('--njobs',
                          required=False,
                          type=int,
                          default=8)
    return arg_pars.parse_args()


def change_data_image(img_path):
    img = cv2.imread(img_path[1], 0)

    if img is None:
        return False

    img = get_actual_area(img)

    cv2.imwrite(img_path[1], img)

    return True


if __name__ == '__main__':
    args = argument_parser()

    data = generate_paths()

    print(data[0])

    res = imap_unordered_bar(change_data_image, data, args.njobs)

    print('Update {} from {}'.format(sum(res), len(data)))


