# -*- coding: utf-8 -*-
# @Author  : LG

import os
import cv2
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
        image = cv2.imread(image_path)
        self.height, self.width, self.depth = image.shape
        del image

        self.objects:List[Object,...] = []

    def load_annotation(self):
        if os.path.exists(self.label_path):
            with open(self.label_path, 'r') as f:
                dataset = load(f)
                info = dataset.get('info', {})
                objects = dataset.get('objects', [])

                self.img_name = info.get('name', '')
                self.width = info.get('width', 0)
                self.height = info.get('height', 0)
                self.depth = info.get('depth', 0)
                self.note = info.get('note', '')
                for obj in objects:
                    category = obj.get('category', 'unknow')
                    group = obj.get('group', '')
                    segmentation = obj.get('segmentation', [])
                    iscrowd = obj.get('iscrowd', 0)
                    note = obj.get('note', '')
                    area = obj.get('area', 0)
                    layer = obj.get('layer', 1)
                    bbox = obj.get('bbox', [])
                    obj = Object(category, group, segmentation, area, layer, bbox, iscrowd, note)
                    self.objects.append(obj)

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
