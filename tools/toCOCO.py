# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5.QtCore import QThread, pyqtSignal
from json import load, dump
import os
import numpy as np
import yaml

class TOCOCO(QThread):
    message = pyqtSignal(int, int, str)

    def __init__(self):
        super(TOCOCO, self).__init__()
        self.isat_json_root:str = None
        self.to_path:str = None
        self.cancel = False

    def run(self):
        coco_anno = {}
        # info
        coco_anno['info'] = {}
        coco_anno['info']['description'] = 'coco from ISAT'
        coco_anno['info']['version'] = None
        coco_anno['info']['year'] = None
        coco_anno['info']['contributor'] = None
        coco_anno['info']['date_created'] = None

        # licenses
        coco_anno['licenses'] = []
        license1 = {}
        license1['url'] = None
        license1['id'] = 0
        license1['name'] = None
        coco_anno['licenses'].append(license1)

        # images and annotations
        coco_anno['images'] = []
        coco_anno['annotations'] = []
        coco_anno['categories'] = []

        categories_dict = {}
        uncontained_dict = {}
        # categories_dict from isat.yaml, https://github.com/yatengLG/ISAT_with_segment_anything/issues/36
        yaml_path = os.path.join(self.isat_json_root, 'isat.yaml')
        if os.path.exists(yaml_path):
            with open(yaml_path, 'rb')as f:
                cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
                for index, label_dict in enumerate(cfg.get('label', [])):
                    label = label_dict.get('name', 'UNKNOW')
                    categories_dict[label] = index

        jsons = [f for f in os.listdir(self.isat_json_root) if f.endswith('.json')]
        num_jsons = len(jsons)
        self.message.emit(None, None, 'Loading ISAT jsons...')
        for file_index, json in enumerate(jsons):
            if self.cancel:
                return
            self.message.emit(file_index+1, num_jsons, '{:>8d}/{:<8d} | Loading ISAT json: {}'.format(file_index+1, num_jsons, json))
            try:
                with open(os.path.join(self.isat_json_root, json), 'r') as f:
                    dataset = load(f)
                    info = dataset.get('info', {})
                    description = info.get('description', '')
                    if not description.startswith('ISAT'):
                        # 不是ISAT格式json
                        continue

                    img_name = info.get('name', '')
                    width = info.get('width', None)
                    height = info.get('height', None)
                    depth = info.get('depth', None)
                    note = info.get('note', '')
                    objects = dataset.get('objects', [])

                    # image
                    coco_image_info = {}
                    coco_image_info['license'] = None
                    coco_image_info['url'] = None
                    coco_image_info['file_name'] = img_name
                    coco_image_info['height'] = height
                    coco_image_info['width'] = width
                    coco_image_info['date_captured'] = None
                    coco_image_info['id'] = file_index
                    coco_anno['images'].append(coco_image_info)

                    objects_groups = [obj.get('group', 0) for obj in objects]
                    objects_groups.sort()
                    objects_groups = set(objects_groups)
                    # 同group
                    for group_index, group in enumerate(objects_groups):
                        objs_with_group = [obj for obj in objects if obj.get('group', 0) == group]
                        cats = [obj.get('category', 'unknow') for obj in objs_with_group]
                        cats = set(cats)
                        # 同category
                        for cat in cats:
                            if cat not in categories_dict:
                                categories_dict[cat] = len(categories_dict)
                                uncontained_dict[cat] = len(categories_dict)
                            category_index = categories_dict.get(cat)

                            objs_with_cat = [obj for obj in objs_with_group if obj.get('category', 0) == cat]
                            crowds = [obj.get('iscrowd', 'unknow') for obj in objs_with_group]
                            crowds = set(crowds)
                            # 同iscrowd
                            for crowd in crowds:
                                objs_with_crowd = [obj for obj in objs_with_cat if obj.get('iscrowd', 0) == crowd]
                                # anno
                                coco_anno_info = {}
                                coco_anno_info['iscrowd'] = crowd
                                coco_anno_info['image_id'] = file_index
                                coco_anno_info['image_name'] = img_name
                                coco_anno_info['category_id'] = category_index
                                coco_anno_info['id'] = len(coco_anno['annotations'])
                                coco_anno_info['segmentation'] = []
                                coco_anno_info['area'] = 0.
                                coco_anno_info['bbox'] = []

                                for obj in objs_with_crowd:

                                    segmentation = obj.get('segmentation', [])
                                    area = obj.get('area', 0)
                                    bbox = obj.get('bbox', [])
                                    if bbox is None:
                                        segmentation_nd = np.array(segmentation)
                                        bbox = [min(segmentation_nd[:, 0]), min(segmentation_nd[:, 1]),
                                                max(segmentation_nd[:, 0]), max(segmentation_nd[:, 1])]
                                        del segmentation_nd
                                    segmentation = [e for p in segmentation for e in p]



                                    if bbox != []:
                                        if coco_anno_info['bbox'] == []:
                                            coco_anno_info['bbox'] = bbox
                                        else:
                                            bbox_tmp = coco_anno_info['bbox']
                                            bbox_tmp = [min(bbox_tmp[0], bbox[0]), min(bbox_tmp[1], bbox[1]),
                                                        max(bbox_tmp[2], bbox[2]), max(bbox_tmp[3], bbox[3])]
                                            coco_anno_info['bbox'] = bbox_tmp
                                    coco_anno_info['segmentation'].append(segmentation)
                                    if area is not None:
                                        coco_anno_info['area'] += float(area)

                                # (xmin, ymin, xmax, ymax) 2 (xmin, ymin, w, h)
                                bbox_tmp = coco_anno_info['bbox']
                                coco_anno_info['bbox'] = [bbox_tmp[0], bbox_tmp[1],
                                                          bbox_tmp[2] - bbox_tmp[0], bbox_tmp[3] - bbox_tmp[1]]

                                coco_anno['annotations'].append(coco_anno_info)
                self.message.emit(None, None, ' ' * 18 + '| Loading finished.')
            except Exception as e:
                self.message.emit(None, None, ' ' * 18 + '| Error: {}'.format(e))

        categories_dict = sorted(categories_dict.items(), key=lambda x:x[1])
        coco_anno['categories'] = [{'name': name, 'id': id, 'supercategory': None} for name, id in categories_dict]

        self.message.emit(None, None, 'Saving COCO json {}'.format(self.to_path))
        with open(self.to_path, 'w') as f:
            try:
                dump(coco_anno, f)
                self.message.emit(None, None, 'Saved finished!')

            except Exception as e:
                self.message.emit(None, None, 'Error: {}'.format(e))

        if uncontained_dict:
            for k,i in uncontained_dict.items():
                self.message.emit(None, None, 'Warning!!! The category [{}] is not contained in isat.yaml. Add it by id {}.'.format(k, i))
        
        self.message.emit(None, None, '*** Finished! ***')

    def __del__(self):
        self.wait()
