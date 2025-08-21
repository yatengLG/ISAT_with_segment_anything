# -*- coding: utf-8 -*-
# @Author  : LG

from ISAT.formats.isat import ISAT
from json import dump, load
import tqdm
import os


class LABELME(ISAT):
    """
    LABELME format

    Attributes:
        keep_crowd (bool): keep the crowded objects
    """
    def __init__(self):
        self.keep_crowd = True

    def read_from_LABELME(self, json_root: str) -> bool:
        """
        Load annotations from the directory of labelme json files.

        Arguments:
            json_root (str): the directory of label json files.
        """
        self.annos.clear()
        self.cates = ()

        json_files = [file for file in os.listdir(json_root) if file.endswith('.json')]
        pbar = tqdm.tqdm(json_files)

        for file in pbar:
            name_without_suffix = self.remove_file_suffix(file)
            pbar.set_description('Load labelme json {}'.format(name_without_suffix+'.json'))
            anno = self.load_one_labelme_json(os.path.join(json_root, file))
            self.annos[name_without_suffix] = anno

        class_set = set()
        for _, anno in self.annos.items():
            for obj in anno.objs:
                class_set.add(obj.category)

        class_set = list(class_set)
        class_set.sort()
        self.cates = tuple(class_set)
        return True

    def save_to_LABELME(self, json_root: str) -> bool:
        """
        Save annotations to the directory of labelme json files.

        Arguments:
            json_root (str): the directory of label json files.
        """
        os.makedirs(json_root, exist_ok=True)

        pbar = tqdm.tqdm(self.annos.items())
        for name_without_suffix, anno in pbar:
            json_path = os.path.join(json_root, name_without_suffix + '.json')
            try:
                self.save_one_labelme_json(anno, json_path)
                pbar.set_description('Save labelme to {}'.format(name_without_suffix+'.json'))

            except Exception as e:
                raise '{} {}'.format(name_without_suffix, e)
        return True

    def save_one_labelme_json(self, anno:ISAT.ANNO, json_path) -> bool:
        """
        Save annotation to a json file.

        Arguments:
            anno (ISAT.ANNO): the annotation.
            json_path (str): the path to save the json file.
        """
        labelme_anno = {}
        labelme_anno['version'] = "5.2.0.post4 | ISAT to LabelMe"
        labelme_anno['imagePath'] = anno.info.name
        labelme_anno['imageData'] = None
        labelme_anno['imageHeight'] = anno.info.height
        labelme_anno['imageWidth'] = anno.info.width
        labelme_anno['flags'] = {}
        labelme_anno['shapes'] = []

        for obj in anno.objs:
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

            shape = {}
            shape['label'] = category
            shape['points'] = segmentation
            shape['group_id'] = int(group) if group else None
            shape['description'] = note
            shape['shape_type'] = 'polygon'
            shape['flags'] = {}

            labelme_anno['shapes'].append(shape)

        with open(json_path, 'w', encoding='utf-8') as f:
            dump(labelme_anno, f, indent=4, ensure_ascii=False)
        return True

    def load_one_labelme_json(self, json_path: str) -> ISAT.ANNO:
        """
        Load annotation from a labelme json file.

        Arguments:
            json_path (str): the path to save the json file.

        Returns:
            ISAT.ANNO: the annotation.
        """
        anno = self.ANNO()
        anno.info = self.ANNO.INFO()

        with open(json_path, 'r', encoding='utf-8') as f:
            dataset = load(f)
            anno.info.description = 'ISAT'
            anno.info.folder = os.path.split(json_path)[0]
            anno.info.name = dataset.get('imagePath')
            anno.info.height = dataset.get('imageHeight')
            anno.info.width = dataset.get('imageWidth')
            anno.info.depth = None
            anno.info.note = ''
            shapes = dataset.get('shapes', {})

            objs = []
            for shape in shapes:
                # 只加载多边形
                is_polygon = shape.get('shape_type', '') == 'polygon'
                if not is_polygon:
                    continue
                obj = self.ANNO.OBJ()

                obj.category = shape.get('label', 'unknow')
                obj.group = shape.get('group_id', 0)
                if obj.group is None: obj.group = 0
                obj.segmentation = shape.get('points', [])
                obj.area = shape.get('area', 0)
                obj.layer = shape.get('layer', 1)
                obj.bbox = shape.get('bbox', [])
                obj.iscrowd = shape.get('iscrowd', False)
                obj.note = shape.get('note', '')

                objs.append(obj)
            anno.objs = tuple(objs)
        return anno