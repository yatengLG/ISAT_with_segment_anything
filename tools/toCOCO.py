# -*- coding: utf-8 -*-
# @Author  : LG

from json import load, dump
import os
from pycocotools import mask as coco_mask
import cv2
import imgviz
import yaml
import numpy as np


class COCOConverter:
    def convert_to_coco(self, isat_json_root:str, to_path:str):
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

        jsons = [f for f in os.listdir(isat_json_root) if f.endswith('.json')]
        for file_index, json in enumerate(jsons):
            print('Load ISAT: {}'.format(json))
            try:
                with open(os.path.join(isat_json_root, json), 'r') as f:
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
            except Exception as e:
                print('Load ISAT: {}, error: {}'.format(json, e))

        categories_dict = sorted(categories_dict.items(), key=lambda x:x[1])
        coco_anno['categories'] = [{'name': name, 'id': id, 'supercategory': None} for name, id in categories_dict]

        with open(to_path, 'w') as f:
            try:
                dump(coco_anno, f)
                print('Save coco json to {}'.format(to_path))
            except Exception as e:
                print('Save {} error :{}'.format(to_path, e))


    def convert_from_coco(self, coco_json_path:str, to_root:str, keep_crowd:bool=False):
        assert coco_json_path.endswith('.json')
        annos = {}
        if os.path.exists(coco_json_path):
            with open(coco_json_path, 'r') as f:
                dataset = load(f)
                images = {image.get('id', None):{
                    'file_name': image.get('file_name', ''),
                    'height': image.get('height', ''),
                    'width': image.get('width', ''),
                } for image in dataset.get('images', [])}
                annotations = dataset.get('annotations', [])
                categories = {categorie.get('id', None): {'name': categorie.get('name', '')} for categorie in dataset.get('categories', [])}
                for index, annotation in enumerate(annotations):

                    annotation_index = annotation.get('id')
                    annotation_image_id = annotation.get('image_id')
                    annotation_category_id = annotation.get('category_id')

                    file_name = images[annotation_image_id].get('file_name')
                    height = images[annotation_image_id].get('height')
                    width = images[annotation_image_id].get('width')
                    iscrowd = annotation["iscrowd"]

                    if file_name == '000000279278.jpg':
                        continue
                    if annotation_image_id not in annos:
                        annos[annotation_image_id] = {}

                    objects = annos[annotation_image_id].get('objects', [])

                    if iscrowd == 0:
                        # polygon
                        segmentations = annotation.get('segmentation')
                        for segmentation in segmentations:
                            xs = segmentation[::2]
                            ys = segmentation[1::2]
                            points = [[x, y] for x ,y in zip(xs, ys)]
                            obj = {
                                'category': categories.get(annotation_category_id).get('name'),
                                'group': annotation_index,
                                'area': None,
                                'segmentation': points,
                                'layer': 1,
                                'bbox': None,
                                'iscrowd': iscrowd,
                                'note': ''
                            }
                            objects.append(obj)
                    elif iscrowd == 1 and keep_crowd:
                        segmentations = annotation.get('segmentation', {})
                        if isinstance(segmentations, dict) and 'counts' in segmentations:
                            # RLE
                            rles = coco_mask.frPyObjects(segmentations, height, width)
                            masks = coco_mask.decode(rles)
                            contours, _ = cv2.findContours(masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
                            for contour in contours:
                                points = []
                                for point in contour:
                                    x, y = point[0]
                                    points.append([float(x), float(y)])
                                obj = {
                                    'category': categories.get(annotation_category_id).get('name'),
                                    'group': annotation_index,
                                    'area': None,
                                    'segmentation': points,
                                    'layer': 1,
                                    'bbox': None,
                                    'iscrowd': iscrowd,
                                    'note': ''
                                }
                                objects.append(obj)
                        else:
                            # polygon
                            for segmentation in segmentations:
                                xs = segmentation[::2]
                                ys = segmentation[1::2]
                                points = [[x, y] for x, y in zip(xs, ys)]
                                obj = {
                                    'category': categories.get(annotation_category_id).get('name'),
                                    'group': annotation_index,
                                    'area': None,
                                    'segmentation': points,
                                    'layer': 1,
                                    'bbox': None,
                                    'iscrowd': iscrowd,
                                    'note': ''
                                }
                                objects.append(obj)
                    else:
                        pass
                    annos[annotation_image_id]['objects'] = objects


                for image_id, values in annos.items():
                    image_path = images[image_id].get('file_name')
                    folder, name = os.path.split(image_path)
                    height = images[image_id].get('height')
                    width = images[image_id].get('width')
                    objects = values.get('objects', [])

                    isat_anno = {}
                    isat_anno['info'] = {}
                    isat_anno['info']['description'] = 'ISAT'
                    isat_anno['info']['folder'] = folder
                    isat_anno['info']['name'] = name
                    isat_anno['info']['width'] = width
                    isat_anno['info']['height'] = height
                    isat_anno['info']['depth'] = None
                    isat_anno['info']['note'] = ''
                    isat_anno['objects'] = []
                    # coco annotation的id 太大了，这里缩一下，每张图片重新开始计数
                    groups_dict = {}
                    for obj in objects:
                        group = obj.get('group', 0)
                        if group not in groups_dict:
                            groups_dict[group] = len(groups_dict)+1
                    for obj in objects:
                        object = {}
                        object['category'] = obj.get('category', '')
                        if 'background' in object['category']:
                            object['group'] = 0
                        else:
                            object['group'] = groups_dict.get(obj.get('group', 0))
                        object['segmentation'] = obj.get('segmentation', [])
                        object['area'] = obj.get('area', None)
                        object['layer'] = obj.get('layer', None)
                        object['bbox'] = obj.get('bbox', None)
                        object['iscrowd'] = obj.get('iscrowd', 0)
                        object['note'] = obj.get('note', '')
                        isat_anno['objects'].append(object)
                    json_name = '.'.join(name.split('.')[:-1]) + '.json'
                    save_json = os.path.join(to_root, json_name)
                    with open(save_json, 'w') as f:
                        try:
                            dump(isat_anno, f)
                            print('Converted coco to ISAT: {}'.format(json_name))

                        except Exception as e:
                            print('Convert coco to ISAT {} ,error: {}'.format(json_name, e))

                ### 类别文件
                cmap = imgviz.label_colormap()
                sorted(categories)
                for index, (k, categorie_dict) in enumerate(categories.items()):
                    r, g, b = cmap[index+1]
                    categorie_dict['color'] = "#{:02x}{:02x}{:02x}".format(r, g, b)
                print(categories)

                s = yaml.dump({'label': list(categories.values())})
                with open(os.path.join(to_root, 'categorys.yaml'), 'w') as f:
                    f.write(s)
