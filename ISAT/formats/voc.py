# -*- coding: utf-8 -*-
# @Author  : LG

from ISAT.formats.isat import ISAT
import tqdm
import numpy as np
from PIL import Image
import imgviz
from skimage.draw.draw import polygon
import os
from typing import Union


class VOC(ISAT):
    """
    VOC format

    Attributes:
        keep_crowd (bool): keep the crowded objects
        is_instance (bool): mark if an object is instance or sematic
    """
    def __init__(self):
        self.keep_crowd = True
        self.is_instance = False

    def save_to_voc(self, png_root: str) -> bool:
        """
        Save annotations to the directory of voc png files.

        Arguments:
            png_root (str): the directory to save the voc png files.
        """
        os.makedirs(png_root, exist_ok=True)

        # cmap
        cmap = imgviz.label_colormap()
        category_index_dict = {}
        with open(os.path.join(png_root, 'classification.txt'), 'w', encoding='utf-8') as f:
            for index, cate in enumerate(self.cates):
                category_index_dict[cate] = index
                f.write('{}\n'.format(cate))

        pbar = tqdm.tqdm(self.annos.items())
        for name_without_suffix, anno in pbar:
            pbar.set_description('Save to {}'.format(name_without_suffix + '.png'))
            png_path = os.path.join(png_root, name_without_suffix + '.png')
            self.save_one_voc_png(anno, png_path, cmap, category_index_dict)
        return True

    def save_one_voc_png(self, anno:ISAT.ANNO, png_path: str, cmap: np.ndarray, category_index_dict=None) -> bool:
        """
        Save annotation to a VOC png file.

        Arguments:
            anno (ISAT.ANNO): the annotation.
            png_path (str): the path of the png file.
            cmap (np.ndarray): color map. shape [n, 3]
            category_index_dict (dict): the category index dict. {index: category}.
        """
        info = anno.info
        objects = anno.objs

        img_name = info.name
        width = info.width
        height = info.height
        depth = info.depth
        note = info.note
        img = np.zeros(shape=(height, width), dtype=np.uint8)

        objects = sorted(objects, key=lambda obj:obj.layer)

        for obj in objects:
            category = obj.category
            group = obj.group
            segmentation = obj.segmentation
            iscrowd = obj.iscrowd
            if iscrowd:
                if not self.keep_crowd:
                    continue
            note = obj.note
            area = obj.area
            layer = obj.layer
            bbox = obj.bbox
            segmentation = [(int(p[1]), int(p[0])) for p in segmentation]

            if self.is_instance and group != '':
                group = int(group)
                assert 0 <= group < 256, 'When use VOC for segmentation, the group must in [0, 255], but get group={}'.format(group)
                self.fill_polygon(segmentation, img, color=group)
            else:
                index = category_index_dict.get(category, 0)
                assert 0 <= index < 256, 'When use VOC for segmentation, the number of classifications must in [0, 255], but get {}'.format(index)
                self.fill_polygon(segmentation, img, color=index)

        img = Image.fromarray(img.astype(np.uint8), mode='P')

        img.putpalette(cmap.flatten())
        img.save(png_path)
        return True

    @staticmethod
    def fill_polygon(segmentation: Union[list, tuple], img: np.ndarray, color: int):
        """
        fill polygon with color on image.

        Arguments:
            segmentation (Union[list, tuple]): the vertices of the polygon. [(x1, y1), (x2, y2), ...] .
            img (np.ndarray): the image.
            color (int): the color of the polygon. save image as mode 'P' with PIL.
        """
        xs = [x for x, y in segmentation]
        ys = [y for x, y in segmentation]
        rr, cc = polygon(xs, ys, img.shape)
        img[rr, cc] = color