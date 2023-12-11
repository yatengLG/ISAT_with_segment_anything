# -*- coding: utf-8 -*-
# @Author  : LG

from ISAT.scripts.isat import ISAT
from pycocotools import coco as mscoco
from pycocotools import mask as mscoco_mask
from json import dump
import cv2
import tqdm
import numpy as np


class COCO(ISAT):
    def __init__(self):
        self.keep_crowd = True

    def read_from_coco(self, annotation_file):
        dataset = mscoco.COCO(annotation_file)
        cats = dataset.cats
        self.cates = [cat.get('name', 'UNKNOW') for _, cat in cats.items()]
        pbar = tqdm.tqdm(dataset.imgs.items())
        for img_id, image in pbar:
            pbar.set_description('Load COCO with image id {}'.format(img_id))
            group = 1
            license = image.get('license', '')
            file_name = image.get('file_name', '')
            coco_url = image.get('license', '')
            height = image.get('height', None)
            width = image.get('width', None)
            date_captured = image.get('date_captured', '')
            flickr_url = image.get('flickr_url', '')
            id = image.get('id', None)

            anno = self.ANNO()
            anno.info = self.ANNO.INFO()
            anno.info.name = file_name
            anno.info.height = height
            anno.info.width = width
            objs = []

            ann_ids = dataset.getAnnIds(img_id)
            for ann_id in ann_ids:
                annos_coco = dataset.loadAnns(ann_id)
                anno_coco = annos_coco[0]

                segmentations = anno_coco.get('segmentation', [])    # 多个polygon
                area = anno_coco.get('area', None)                   # coco中，area是组面积，isat中是单个polygon面积
                iscrowd = anno_coco.get('iscrowd', None)
                image_id = anno_coco.get('image_id', None)
                bbox = anno_coco.get('bbox', [])
                category_id = anno_coco.get('category_id', None)
                id = anno_coco.get('id', None)

                if iscrowd == 0:
                    # polygon
                    for segmentation in segmentations:
                        xs = segmentation[::2]
                        ys = segmentation[1::2]
                        points = [[x, y] for x, y in zip(xs, ys)]

                        obj = self.ANNO.OBJ()
                        obj.category = dataset.loadCats(category_id)[0].get('name', 'UNKNOW')
                        obj.group = group
                        obj.area = None
                        obj.segmentation = points
                        obj.layer = 1
                        obj.bbox = None
                        obj.iscrowd = iscrowd
                        obj.note = ''

                        objs.append(obj)

                elif iscrowd == 1 and self.keep_crowd:
                    if isinstance(segmentations, dict) and 'counts' in segmentations:
                        # RLE
                        rles = mscoco_mask.frPyObjects(segmentations, height, width)
                        masks = mscoco_mask.decode(rles)
                        contours, _ = cv2.findContours(masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
                        for contour in contours:
                            points = []
                            for point in contour:
                                x, y = point[0]
                                points.append([float(x), float(y)])
                            obj = self.ANNO.OBJ()
                            obj.category = dataset.loadCats(category_id)[0].get('name', 'UNKNOW')
                            obj.group = group
                            obj.area = None
                            obj.segmentation = points
                            obj.layer = 1
                            obj.bbox = None
                            obj.iscrowd = iscrowd
                            obj.note = ''

                            objs.append(obj)

                    else:
                        # polygon
                        for segmentation in segmentations:
                            xs = segmentation[::2]
                            ys = segmentation[1::2]
                            points = [[x, y] for x, y in zip(xs, ys)]
                            obj = self.ANNO.OBJ()
                            obj.category = dataset.loadCats(category_id)[0].get('name', 'UNKNOW')
                            obj.group = group
                            obj.area = None
                            obj.segmentation = points
                            obj.layer = 1
                            obj.bbox = None
                            obj.iscrowd = iscrowd
                            obj.note = ''

                            objs.append(obj)

                else:
                    pass
                group += 1
            anno.objs = tuple(objs)
            self.annos[self.remove_file_suffix(file_name)] = anno
        return True

    def save_to_coco(self, annotation_file):

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

        pbar = tqdm.tqdm(self.annos.items())
        for file_index, (name_without_suffix, anno) in enumerate(pbar):
            pbar.set_description('Integrate {}'.format(name_without_suffix))
            try:
                if not anno.info.description.startswith('ISAT'):
                    continue

                # image
                coco_image_info = {}
                coco_image_info['license'] = ''
                coco_image_info['url'] = ''
                coco_image_info['file_name'] = anno.info.name
                coco_image_info['height'] = anno.info.height
                coco_image_info['width'] = anno.info.width
                coco_image_info['date_captured'] = ''
                coco_image_info['id'] = file_index
                coco_anno['images'].append(coco_image_info)

                objects = anno.objs

                objects_groups = [obj.group for obj in objects]
                objects_groups.sort()
                objects_groups = set(objects_groups)
                # 同group
                for group_index, group in enumerate(objects_groups):
                    objs_with_group = [obj for obj in objects if obj.group == group]
                    cats = [obj.category for obj in objs_with_group]
                    cats = set(cats)
                    # 同category
                    for cat in cats:
                        if cat not in categories_dict:
                            categories_dict[cat] = len(categories_dict)
                            uncontained_dict[cat] = len(categories_dict)
                        category_index = categories_dict.get(cat)

                        objs_with_cat = [obj for obj in objs_with_group if obj.category == cat]
                        crowds = [obj.iscrowd for obj in objs_with_group]
                        crowds = set(crowds)
                        # 同iscrowd
                        for crowd in crowds:
                            objs_with_crowd = [obj for obj in objs_with_cat if obj.iscrowd == crowd]
                            # anno
                            coco_anno_info = {}
                            coco_anno_info['iscrowd'] = crowd
                            coco_anno_info['image_id'] = file_index
                            coco_anno_info['image_name'] = anno.info.name
                            coco_anno_info['category_id'] = category_index
                            coco_anno_info['id'] = len(coco_anno['annotations'])
                            coco_anno_info['segmentation'] = []
                            coco_anno_info['area'] = 0.
                            coco_anno_info['bbox'] = []

                            for obj in objs_with_crowd:

                                segmentation = obj.segmentation
                                area = obj.area
                                bbox = obj.bbox
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
            except Exception as e:
                raise '{} {}'.format(name_without_suffix, e)

        categories_dict = sorted(categories_dict.items(), key=lambda x: x[1])
        coco_anno['categories'] = [{'name': name, 'id': id, 'supercategory': None} for name, id in categories_dict]

        with open(annotation_file, 'w') as f:
            try:
                dump(coco_anno, f)
                print('Save COCO json finished.')
            except Exception as e:
                print('Save COCO json error: {}'.format(e))

        return True

