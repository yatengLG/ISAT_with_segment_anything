# -*- coding: utf-8 -*-
# @Author  : LG

import os
from PIL import Image
import numpy as np
from json import load, dump
from typing import List


class Object:
    def __init__(self, category:str, group:int, segmentation, area, layer, bbox, iscrowd=0, note=''):
        self.category = category
        self.group = group
        self.segmentation = segmentation
        self.area = area
        self.layer = layer
        self.bbox = bbox
        self.iscrowd = iscrowd
        self.note = note


class Annotation:
    def __init__(self, image_path, label_path):
        img_folder, img_name = os.path.split(image_path)
        self.description = 'ISAT'
        self.img_folder = img_folder
        self.img_name = img_name
        self.label_path = label_path
        self.note = ''

        image = np.array(Image.open(image_path))
        if image.ndim == 3:
            self.height, self.width, self.depth = image.shape
        elif image.ndim == 2:
            self.height, self.width = image.shape
            self.depth = 0
        else:
            self.height, self.width, self.depth = image.shape[:, :3]
            print('Warning: Except image has 2 or 3 ndim, but get {}.'.format(image.ndim))
        del image

        self.objects:List[Object,...] = []

    def load_annotation(self):
        if os.path.exists(self.label_path):
            with open(self.label_path, 'r') as f:
                dataset = load(f)
                info = dataset.get('info', {})
                description = info.get('description', '')
                if description == 'ISAT':
                    # ISAT格式json
                    objects = dataset.get('objects', [])
                    self.img_name = info.get('name', '')
                    width = info.get('width', None)
                    if width is not None:
                        self.width = width
                    height = info.get('height', None)
                    if height is not None:
                        self.height = height
                    depth = info.get('depth', None)
                    if depth is not None:
                        self.depth = depth
                    self.note = info.get('note', '')
                    for obj in objects:
                        category = obj.get('category', 'unknow')
                        group = obj.get('group', 0)
                        if group is None: group = 0
                        segmentation = obj.get('segmentation', [])
                        iscrowd = obj.get('iscrowd', 0)
                        note = obj.get('note', '')
                        area = obj.get('area', 0)
                        layer = obj.get('layer', 2)
                        bbox = obj.get('bbox', [])
                        obj = Object(category, group, segmentation, area, layer, bbox, iscrowd, note)
                        self.objects.append(obj)
                else:
                    # 不再支持直接打开labelme标注文件（在菜单栏-tool-convert中提供了isat<->labelme相互转换工具）
                    print('Warning: The file {} is not a ISAT json.'.format(self.label_path))
        return self

    def save_annotation(self):
        dataset = {}
        dataset['info'] = {}
        dataset['info']['description'] = self.description
        dataset['info']['folder'] = self.img_folder
        dataset['info']['name'] = self.img_name
        dataset['info']['width'] = self.width
        dataset['info']['height'] = self.height
        dataset['info']['depth'] = self.depth
        dataset['info']['note'] = self.note
        dataset['objects'] = []
        for obj in self.objects:
            object = {}
            object['category'] = obj.category
            object['group'] = obj.group
            object['segmentation'] = obj.segmentation
            object['area'] = obj.area
            object['layer'] = obj.layer
            object['bbox'] = obj.bbox
            object['iscrowd'] = obj.iscrowd
            object['note'] = obj.note
            dataset['objects'].append(object)
        with open(self.label_path, 'w') as f:
            dump(dataset, f, indent=4)
        return True
