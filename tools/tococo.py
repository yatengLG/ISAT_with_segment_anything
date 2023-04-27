# -*- coding: utf-8 -*-
# @Author  : LG

from json import load, dump
import os
from pycocotools import mask as coco_mask
import cv2
import imgviz


class COCOConverter:
    def __init__(self, ):
        pass
        # labels = cfg.get('label', [])
        # for index, label_dict in enumerate(labels):
        #     category = label_dict.get('name', 'unknow')
        #     color = label_dict.get('color', '#000000')

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
        coco_anno['licenses'][0] = {}
        coco_anno['licenses'][0]['url'] = None
        coco_anno['licenses'][0]['id'] = 0
        coco_anno['licenses'][0]['name'] = None

        # images and annotations
        coco_anno['images'] = []
        coco_anno['annotations'] = []

        jsons = [f for f in os.listdir(isat_json_root) if f.endswith('.json')]
        for json in jsons:
            coco_image_info = {}

            dataset = load(os.path.join(isat_json_root, json))
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



    def convert_from_coco(self, coco_json:str, to_path:str, keep_crowd:bool=False):
        assert coco_json.endswith('.json')
        annos = {}
        if os.path.exists(coco_json):
            with open(coco_json, 'r') as f:
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
                        # RLE
                        segmentations = annotation.get('segmentation', {})
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
                        object['group'] = groups_dict.get(obj.get('group', 0))
                        object['segmentation'] = obj.get('segmentation', [])
                        object['area'] = obj.get('area', None)
                        object['layer'] = obj.get('layer', None)
                        object['bbox'] = obj.get('bbox', None)
                        object['iscrowd'] = obj.get('iscrowd', 0)
                        object['note'] = obj.get('note', '')
                        isat_anno['objects'].append(object)
                    json_name = '.'.join(name.split('.')[:-1]) + '.json'
                    save_json = os.path.join(to_path, json_name)
                    with open(save_json, 'w') as f:
                        try:
                            dump(isat_anno, f)
                            print('Converted coco to ISAT: {}'.format(json_name))

                        except Exception as e:
                            print('Convert coco to ISAT {} ,error: {}'.format(json_name, e))

                    ### 类别文件
                    cmap = imgviz.label_colormap()


if __name__ == '__main__':
    coco = COCOConverter()
    coco.convert_from_coco(
        '/mnt/disk/coco/annotations/instances_val2017.json',
        '/mnt/disk/coco/isat_json',
        True
    )