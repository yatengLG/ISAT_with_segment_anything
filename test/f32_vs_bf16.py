# -*- coding: utf-8 -*-
# @Author  : LG

from ISAT.segment_any.segment_any import SegAny
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time
import os


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_res(masks, input_point, input_label, filename, image):
    for i, mask in enumerate(masks):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if (input_point is not None) and (input_label is not None):
            show_points(input_point, input_label, plt.gca())

        plt.axis('off')
        plt.savefig(filename + '_' + str(i) + '.png', bbox_inches='tight', pad_inches=-0.1)
        plt.close()


images = {0: {'path': '../example/images/000000000113.jpg',
              'point': [(113, 207)],
              'label': [1]},
          1: {'path': '../example/images/000000000144.jpg',
              'point': [(78, 287)],
              'label': [1]},
          2: {'path': '../example/images/000000000308.jpg',
              'point': [(496, 125), (490, 295)],
              'label': [1, 1]},
          3: {'path': '../example/images/000000000872.jpg',
              'point': [(352, 205), (192, 485), (267, 330)],
              'label': [1, 1, 0]},
          4: {'path': '../example/images/000000002592.jpg',
              'point': [(309, 98)],
              'label': [1]},
          5: {'path': '../example/images/000000117425.jpg',
              'point': [(407, 213), (392, 347)],
              'label': [1, 1]},
          }

if __name__ == '__main__':
    result_path = 'result/'
    os.makedirs(result_path, exist_ok=True)
    checkpoint_root = '../ISAT/checkpoints'
    record = {}

    for checkpoint in os.listdir(checkpoint_root):
        if not (checkpoint.endswith('.pth') or checkpoint.endswith('.pt')):
            continue

        checkpoint_path = os.path.join(checkpoint_root, checkpoint)
        checkpoint_name = os.path.split(checkpoint)[-1].split('.')[0]

        record[checkpoint] = []
        record[checkpoint].append('x')

        model = SegAny(checkpoint_path, use_bfloat16=False)

        for index, image_info in images.items():
            image_path = image_info['path']
            input_point = np.array(image_info['point'])
            input_label = np.array(image_info['label'])

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            start_time = time.time()
            model.set_image(image)
            masks = model.predict_with_point_prompt(input_point, input_label)
            record[checkpoint].append(time.time() - start_time)
            show_res(masks, input_point, input_label, result_path + "{}_{}_x".format(index, checkpoint_name), image)

        #
        record[checkpoint].append('y')

        model = SegAny(checkpoint_path, use_bfloat16=True)

        for index, image_info in images.items():
            image_path = image_info['path']
            input_point = np.array(image_info['point'])
            input_label = np.array(image_info['label'])

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            start_time = time.time()
            model.set_image(image)
            masks = model.predict_with_point_prompt(input_point, input_label)
            record[checkpoint].append(time.time() - start_time)
            show_res(masks, input_point, input_label, result_path + "{}_{}_y".format(index, checkpoint_name), image)

    #
    print('| {:^40s} | {:^10s} | {:^6s} | {:^6s} | {:^6s} | {:^6s} | {:^6s} | {:^6s} |'.format(
            'checkpoint', 'use_bfloat', '0', '1', '2', '3', '4', '5'))

    for k, v in record.items():
        print('| {:^40s} | {:^10s} | {:5.4f} | {:5.4f} | {:5.4f} | {:5.4f} | {:5.4f} | {:5.4f} |'.format(k, *v[:7]))
        print('| {:^40s} | {:^10s} | {:5.4f} | {:5.4f} | {:5.4f} | {:5.4f} | {:5.4f} | {:5.4f} |'.format(k, *v[7:]))
