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


    def save_to_voc(self, png_root):
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
        # img = np.zeros(shape=(height, width), dtype=np.uint8)
        img = np.full(shape=(height, width), fill_value=255, dtype=np.uint8)

        objects = sorted(objects, key=lambda obj:obj.layer)

        if self.instance_id:
            ins_img = np.full(shape=(height, width), fill_value=0, dtype=np.uint8)
        instance_num_overflow = False

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

            if self.instance_id and group != '':
                group = int(group)
                assert 0 <= group, 'The group must larger than -1, but get group={}'.format(group)
                if not instance_num_overflow and group > 255:
                    instance_num_overflow = True
                    print("Group larger than 255, use uint16")
                    ins_img = ins_img.astype(np.uint16)
                # assert 0 <= group < 256, 'When use VOC for segmentation, the group must in [0, 255], but get group={}'.format(group)
                self.fill_polygon(segmentation, ins_img, color=group)
            
            index = category_index_dict.get(category, 0)
            assert 0 <= index < 256, 'When use VOC for segmentation, the number of classifications must in [0, 255], but get {}'.format(index)
            self.fill_polygon(segmentation, img, color=index)

        if self.instance_id:
            if instance_num_overflow:
                ins_img = Image.fromarray(ins_img.astype(np.uint16), mode='I;16')
            else:
                ins_img = Image.fromarray(ins_img.astype(np.uint8), mode='L')
            ins_img.save(png_path[:-4] + "_instanceId.png")
            ins_img = Image.fromarray(np.array(ins_img, dtype=np.uint8), mode='P')
            ins_img.putpalette(cmap.flatten())
            ins_img.save(png_path)
        
        if self.semantic_id:
            sem_img = Image.fromarray(img.astype(np.uint8), mode='L')
            sem_img.save(png_path[:-4] + "_semanticId.png")

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
