# -*- coding: utf-8 -*-
# @Author  : LG

from json import load
import numpy as np
from PIL import Image, ImageDraw, ImageColor
import mahotas
import imgviz

class Converter:
    def __init__(self, cfg, is_segmentation):
        self.is_segmentation = is_segmentation

        labels = cfg.get('label', [])
        self.category_dict = {}
        if self.is_segmentation:
            self.cmap = imgviz.label_colormap()
        else:
            self.cmap = np.zeros((len(labels), 3), dtype=np.uint8)
            for index, label_dict in enumerate(labels):
                category = label_dict.get('name', 'unknow')
                color = label_dict.get('color', '#000000')
                self.category_dict[category] = {'index': index, 'color': color}
                self.cmap[index] = (ImageColor.getrgb(color))

    def convert(self, from_path:str, to_path:str):
        assert from_path.endswith('.json')

        with open(from_path, 'r') as f:
            dataset = load(f)
            info = dataset.get('info', {})
            objects = dataset.get('objects', [])

            img_name = info.get('name', '')
            width = info.get('width', 0)
            height = info.get('height', 0)
            depth = info.get('depth', 0)
            note = info.get('note', '')
            img = np.zeros(shape=(height, width), dtype=np.uint8)

            objects = sorted(objects, key=lambda obj:obj.get('layer', 1))

            for obj in objects:
                category = obj.get('category', 'unknow')
                group = obj.get('group', '')
                segmentation = obj.get('segmentation', [])
                iscrowd = obj.get('iscrowd', 0)
                note = obj.get('note', '')
                area = obj.get('area', 0)
                layer = obj.get('layer', 1)
                bbox = obj.get('bbox', [])
                segmentation = [(p[1], p[0]) for p in segmentation]

                if self.is_segmentation and group != '':
                    mahotas.polygon.fill_polygon(segmentation, img, color=int(group))
                else:
                    mahotas.polygon.fill_polygon(segmentation, img, color=self.category_dict.get(category, {}).get('index', 0))
        img = Image.fromarray(img.astype(np.uint8), mode='P')
        img.putpalette(self.cmap.flatten())
        img.save(to_path)
        return True

