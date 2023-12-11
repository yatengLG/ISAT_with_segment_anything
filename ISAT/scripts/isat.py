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

    annos:Dict[str, ANNO] = {}  # name, ANNO (the name without the suffix)
    cates:Tuple[str] = ()

    def read_from_ISAT(self, json_root):
        if os.path.exists(os.path.join(json_root, 'isat.yaml')):
            cates = []
            with open(os.path.join(json_root, 'isat.yaml'), 'rb')as f:
                cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
            for label in cfg.get('label', []):
                cates.append(label.get('name'))
            self.cates = tuple(cates)

        pbar = tqdm.tqdm([file for file in os.listdir(json_root) if file.endswith('.json')])
        for file in pbar:
            pbar.set_description('Load ISAT from {}'.format(file))
            anno = self._load_one_isat_json(os.path.join(json_root, file))
            self.annos[self.remove_file_suffix(file)] = anno
        return True

    def save_to_ISAT(self, json_root):
        os.makedirs(json_root, exist_ok=True)

        pbar = tqdm.tqdm(self.annos.items())
        for name_without_suffix, Anno in pbar:
            json_name = name_without_suffix + '.json'
            pbar.set_description('Save ISAT to {}'.format(json_name))
            self._save_one_isat_json(Anno, os.path.join(json_root, json_name))

        # 类别文件
        cmap = imgviz.label_colormap()
        self.cates = sorted(self.cates)
        categories = []
        for index, cat in enumerate(self.cates):
            r, g, b = cmap[index + 1]
            categories.append({
                'name': cat if isinstance(cat, str) else str(cat),
                'color': "#{:02x}{:02x}{:02x}".format(r, g, b)
            })
        s = yaml.dump({'label': categories})
        with open(os.path.join(json_root, 'isat.yaml'), 'w') as f:
            f.write(s)

        return True

    def remove_file_suffix(self, file_name):
        return os.path.splitext(file_name)[0]

    def _load_one_isat_json(self, json_path) -> ANNO:
        anno = self.ANNO()
        with open(json_path, 'r') as f:
            dataset = load(f)
            info = dataset.get('info', {})
            description = info.get('description', '')
            if description != 'ISAT':
                raise AttributeError('The json file {} is`t a ISAT json.'.format(json_path))
            folder = info.get('folder', '')
            img_name = info.get('name', '')
            width = info.get('width', None)
            height = info.get('height', None)
            depth = info.get('depth', None)
            note = info.get('note', '')

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
        return anno

    def _save_one_isat_json(self, anno:ANNO, save_path):
        anno.info.description = 'ISAT'
        dataset = {}
        dataset['info'] = {}
        dataset['info']['description'] = anno.info.description
        dataset['info']['folder'] = anno.info.folder
        dataset['info']['name'] = anno.info.name
        dataset['info']['width'] = anno.info.width
        dataset['info']['height'] = anno.info.height
        dataset['info']['depth'] = anno.info.depth
        dataset['info']['note'] = anno.info.note
        dataset['objects'] = []
        for obj in anno.objs:
            object = {}
            object['category'] = obj.category if isinstance(obj.category, str) else str(obj.category)
            object['group'] = obj.group
            object['segmentation'] = obj.segmentation
            object['area'] = obj.area
            object['layer'] = obj.layer
            object['bbox'] = obj.bbox
            object['iscrowd'] = obj.iscrowd
            object['note'] = obj.note
            dataset['objects'].append(object)

        with open(save_path, 'w') as f:
            dump(dataset, f, indent=4)
        return True