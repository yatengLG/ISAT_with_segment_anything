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
    The ISAT format provides a structured approach for representing image annotations.

    File Naming: Each image has a corresponding .json file named after the image file (without the image extension).

    Attributes:
        annos (dict): Dictionary of image annotations. {name: ANNO}
        cates (tuple): categories.

    """
    class ANNO:
        r"""Annotation class.

        Attributes:
            info (INFO): INFO class.
            objs (tuple[OBJ, ...]): tuple of OBJs.

        """
        class INFO:
            r"""Contains metadata about the dataset and image.

            Attributes:
                description (str): Always "ISAT".
                folder (str): The directory where the images are stored.
                name (str): The name of the image file
                width (int): The dimensions of the image.
                height (int): The dimensions of the image.
                depth (int): The dimensions of the image; depth is assumed to be 3 for RGB images.
                note: An optional field for any additional notes related to the image.
            """
            description = ''
            folder = ''
            name = ''
            width = None
            height = None
            depth = None
            note = ''
        class OBJ:
            r"""Lists all the annotated objects in the image.

            Attributes:
                category (str): The class label of the object.
                group (int): An identifier that groups objects based on overlapping bounding boxes. If an object's bounding box is within another, they share the same group number.
                segmentation (list | tuple): A list of [x, y] coordinates forming the polygon around the object.
                area (float): The area covered by the object in pixels.
                layer (int): A float indicating the sequence of the object. It increments within the same group, starting at 1.0 .
                bbox (list | tuple): The bounding box coordinates in the format [x_min, y_min, x_max, y_max].
                iscrowd (bool): A boolean value indicating if the object is part of a crowd.
                note (str): An optional field for any additional notes related to the object.
            """
            category = ''
            group = None
            segmentation = None
            area = None
            layer = None
            bbox = None
            iscrowd = False
            note = ''
        info:INFO
        objs:Tuple[OBJ, ...] = ()

    annos:Dict[str, ANNO] = {}  # name, ANNO (the name without the suffix)
    cates:Tuple[str] = ()

    def read_from_ISAT(self, json_root: str) -> bool:
        r"""Load annotations from a directory of json files.

        Arguments:
            json_root (str): The directory of json files.
        """
        self.annos.clear()
        self.cates = ()

        if os.path.exists(os.path.join(json_root, 'isat.yaml')):
            cates = []
            with open(os.path.join(json_root, 'isat.yaml'), 'r', encoding='utf-8')as f:
                cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
            for label in cfg.get('label', []):
                cates.append(label.get('name'))
            self.cates = tuple(cates)

        pbar = tqdm.tqdm([file for file in os.listdir(json_root) if file.endswith('.json')])
        for file in pbar:
            pbar.set_description('Load ISAT from {}'.format(file))
            anno = self.load_one_isat_json(os.path.join(json_root, file))
            self.annos[self.remove_file_suffix(file)] = anno
        return True

    def save_to_ISAT(self, json_root: str) -> bool:
        r"""
        Save annotations to the directory of json files.

        Arguments:
            json_root (str): The directory of json files.
        """
        os.makedirs(json_root, exist_ok=True)

        pbar = tqdm.tqdm(self.annos.items())
        for name_without_suffix, Anno in pbar:
            json_name = name_without_suffix + '.json'
            pbar.set_description('Save ISAT to {}'.format(json_name))
            self.save_one_isat_json(Anno, os.path.join(json_root, json_name))

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
        s = yaml.dump({'label': categories}, allow_unicode=True)
        with open(os.path.join(json_root, 'isat.yaml'), 'w', encoding='utf-8') as f:
            f.write(s)

        return True

    def remove_file_suffix(self, file_name: str) -> str:
        r"""
        Remove the file suffix from the file name.

        Arguments:
            file_name (str): The file name.
        Returns:
            str: The file name without the file suffix.
        """
        return os.path.splitext(file_name)[0]

    def load_one_isat_json(self, json_path: str) -> ANNO:
        r"""
        Load annotation from a json file.

        Arguments:
            json_path (str): The file path.
        Returns:
            ANNO: The instance of the ANNO.
        """
        anno = self.ANNO()
        with open(json_path, 'r', encoding='utf-8') as f:
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
                iscrowd = obj.get('iscrowd', False)
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

    def save_one_isat_json(self, anno:ANNO, save_path:str) -> bool:
        r"""
        Save annotation to a json file.

        Arguments:
            anno (ANNO): The instance of the ANNO.
            save_path (str): The ISAT json file path.
        """
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

        with open(save_path, 'w', encoding='utf-8') as f:
            dump(dataset, f, indent=4, ensure_ascii=False)
        return True