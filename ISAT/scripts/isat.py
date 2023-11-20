# -*- coding: utf-8 -*-
# @Author  : LG

from typing import Dict, Tuple
from json import dump, load
import yaml
import imgviz
import tqdm
import os


class ISAT:
    """
    The ISAT (Image Segmentation Annotation Tool) format provides a structured approach for representing image annotations
    File Naming: Each image has a corresponding .json file named after the image file (without the image extension)
        ['info']: Contains metadata about the dataset and image
            ['description']: Always 'ISAT'
            ['folder']: The directory where the images are stored
            ['name']: The name of the image file
            ['width'], ['height'], ['depth']: The dimensions of the image; depth is assumed to be 3 for RGB images
            ['note']: An optional field for any additional notes related to the image
        ['objects']: Lists all the annotated objects in the image
            ['category']: The class label of the object.
            ['group']: An identifier that groups objects based on overlapping bounding boxes. If an object's bounding box is within another, they share the same group number.
            ['segmentation']: A list of [x, y] coordinates forming the polygon around the object
            ['area']: The area covered by the object in pixels
            ['layer']: A float indicating the sequence of the object. It increments within the same group, starting at 1.0
            ['bbox']: The bounding box coordinates in the format [x_min, y_min, x_max, y_max]
            ['iscrowd']: A boolean value indicating if the object is part of a crowd
            ['note']: An optional field for any additional notes related to the object
    """
    class ANNO:
        class INFO:
            description = ''
            folder = ''
            name = ''
            width = None
            height = None
            depth = None
            note = ''
        class OBJ:
            category = ''
            group = None
            segmentation = None
            area = None
            layer = None
            bbox = None
            iscrowd = None
            note = ''
        info:INFO
        objs:Tuple[OBJ] = ()

    annos:Dict[str, ANNO] = {}  # name, ANNO
    cates:Tuple[str] = ()

    def read_from_ISAT(self, json_root):
        pbar = tqdm.tqdm([file for file in os.listdir(json_root) if file.endswith('.json')])
        for file in pbar:
            pbar.set_description('Load ISAT from {}'.format(file))
            with open(os.path.join(json_root, file), 'r') as f:
                dataset = load(f)
                info = dataset.get('info', {})
                description = info.get('description', '')
                if description != 'ISAT':
                    continue
                folder = info.get('folder', '')
                img_name = info.get('name', '')
                width = info.get('width', None)
                height = info.get('height', None)
                depth = info.get('depth', None)
                note = info.get('note', '')

                anno = self.ANNO()
                anno.info = self.ANNO.INFO()
                anno.info.description = description
                anno.info.folder = folder
                anno.info.name = img_name
                anno.info.width = width
                anno.info.height = height
                anno.info.depth = depth
                anno.info.note = note

                objs = []
                objects = dataset.get('objects', [])
                for obj in objects:
                    category = obj.get('category', 'UNKNOW')
                    group = obj.get('group', 0)
                    if group is None: group = 0
                    segmentation = obj.get('segmentation', [])
                    iscrowd = obj.get('iscrowd', 0)
                    note = obj.get('note', '')
                    area = obj.get('area', 0)
                    layer = obj.get('layer', 2)
                    bbox = obj.get('bbox', [])

                    obj = self.ANNO.OBJ()
                    obj.category = category
                    obj.group = group
                    obj.segmentation = segmentation
                    obj.area = area
                    obj.layer = layer
                    obj.bbox = bbox
                    obj.iscrowd = iscrowd
                    obj.note = note
                    objs.append(obj)

                anno.objs = tuple(objs)
                self.annos[img_name] = anno
        return self

    def save_to_ISAT(self, json_root):

        pbar = tqdm.tqdm(self.annos.items())
        for name, Anno in pbar:
            json_name = os.path.splitext(Anno.info.name)[0] + '.json'
            pbar.set_description('Save ISAT to {}'.format(json_name))
            Anno.info.description = 'ISAT'
            dataset = {}
            dataset['info'] = {}
            dataset['info']['description'] = Anno.info.description
            dataset['info']['folder'] = Anno.info.folder
            dataset['info']['name'] = Anno.info.name
            dataset['info']['width'] = Anno.info.width
            dataset['info']['height'] = Anno.info.height
            dataset['info']['depth'] = Anno.info.depth
            dataset['info']['note'] = Anno.info.note
            dataset['objects'] = []
            for obj in Anno.objs:
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
            with open(os.path.join(json_root, json_name), 'w') as f:
                dump(dataset, f, indent=4)

        # 类别文件
        cmap = imgviz.label_colormap()
        self.Cates = sorted(self.Cates)
        categories = []
        for index, cat in enumerate(self.Cates):
            r, g, b = cmap[index + 1]
            categories.append({
                'name': cat,
                'color': "#{:02x}{:02x}{:02x}".format(r, g, b)
            })
        s = yaml.dump({'label': categories})
        with open(os.path.join(json_root, 'isat.yaml'), 'w') as f:
            f.write(s)

        return self
