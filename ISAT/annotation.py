# -*- coding: utf-8 -*-
# @Author  : LG

import os
from PIL import Image
import numpy as np
from json import load, dump
from typing import List, Union
from ISAT.utils.dicom import load_dcm_as_image

__all__ = ['Object', 'Annotation']

class Object:
    r"""A class to represent an annotation object.

    Arguments:
        category (str): The category of the object.
        group (int): The group of the object.
        segmentation (list | tuple): The vertices of the object.[(x1, y1), (x2, y2), ...]
        area (float): The area of the object.
        layer (int): The layer of the object.
        bbox (list | tuple): The bbox of the object. [xmin, ymin, xmax, ymax]
        iscrowd (bool): The crowd tag of the object.
        note (str): The note of the object.
    """
    def __init__(self, category: str, group: int, segmentation: Union[list, tuple], area: float, layer: int, bbox: Union[list, tuple], iscrowd: bool=False, note: str=''):
        self.category = category
        self.group = group
        self.segmentation = segmentation
        self.area = area
        self.layer = layer
        self.bbox = bbox
        self.iscrowd = iscrowd
        self.note = note


class Annotation:
    r"""A class to represent an annotation containing many objects.

    Arguments:
        image_path (str): The path to the image.
        label_path (str): The path to the label file.

    Attributes:
        description (str): Always 'ISAT'.
        img_folder (str): The path to the folder where the images are located.
        img_name (str): The name of the image.
        label_path (str): The path to the label file.
        note (str): The note of the image.
        height (int): The height of the image.
        width (int): The width of the image.
        depth (int): The depth of the image.
    """
    def __init__(self, image_path:str, label_path:str):
        img_folder, img_name = os.path.split(image_path)
        self.description = 'ISAT'
        self.img_folder = img_folder
        self.img_name = img_name
        self.label_path = label_path
        self.note = ''

        if image_path.lower().endswith('.dcm'):
            image = np.array(load_dcm_as_image(image_path))
        else:
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

        self.objects:List[Object, ] = []

    def load_annotation(self):
        r"""
        Load annotation from self.label_path
        """
        if os.path.exists(self.label_path):
            with open(self.label_path, 'r', encoding='utf-8') as f:
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
                        iscrowd = obj.get('iscrowd', False)
                        iscrowd = iscrowd if isinstance(iscrowd, bool) else bool(iscrowd)
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
        r"""
        Save annotation to self.label_path
        """
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
        with open(self.label_path, 'w', encoding='utf-8') as f:
            dump(dataset, f, indent=4, ensure_ascii=False)
        return True
