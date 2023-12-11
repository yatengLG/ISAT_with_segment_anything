# -*- coding: utf-8 -*-
# @Author  : LG

from ISAT.scripts.isat import ISAT
import tqdm
import numpy as np
from PIL import Image
import imgviz
from skimage.draw.draw import polygon
import os


class VOC(ISAT):
    def __init__(self):
        self.keep_crowd = True
        self.is_instance = False


    def save_to_voc(self, png_root):
        os.makedirs(png_root, exist_ok=True)

        # cmap
        cmap = imgviz.label_colormap()
        category_index_dict = {}
        with open(os.path.join(png_root, 'classification.txt'), 'w') as f:
            for index, cate in enumerate(self.cates):
                category_index_dict[cate] = index
                f.write('{}\n'.format(cate))

        pbar = tqdm.tqdm(self.annos.items())
        for name_without_suffix, anno in pbar:
            pbar.set_description('Save to {}'.format(name_without_suffix + '.png'))
            png_path = os.path.join(png_root, name_without_suffix + '.png')
            self._save_one_voc_png(anno, png_path, cmap, category_index_dict)

        return True


    def _save_one_voc_png(self, anno:ISAT.ANNO, png_path, cmap, category_index_dict=None):
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
                self.fill_polygon(segmentation, img, color=int(group))
            else:
                self.fill_polygon(segmentation, img, color=category_index_dict.get(category, 0))

        img = Image.fromarray(img.astype(np.uint8), mode='P')

        img.putpalette(cmap.flatten())
        img.save(png_path)
        return True

    @staticmethod
    def fill_polygon(segmentation, img: np.ndarray, color: int):
        xs = [x for x, y in segmentation]
        ys = [y for x, y in segmentation]
        rr, cc = polygon(xs, ys, img.shape)
        img[rr, cc] = color