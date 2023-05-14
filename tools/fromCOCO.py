# -*- coding: utf-8 -*-
# @Author  : LG


from PyQt5.QtCore import QThread, pyqtSignal
from json import load, dump
import os
from pycocotools import mask as coco_mask
import cv2
import imgviz
import yaml

class FROMCOCO(QThread):
    message = pyqtSignal(int, int, str)

    def __init__(self):
        super(FROMCOCO, self).__init__()
        self.coco_json_path:str = None
        self.to_root:str = None
        self.keep_crowd = False

        self.cancel = False

    def run(self):
        assert self.coco_json_path.endswith('.json')
        annos = {}
        if os.path.exists(self.coco_json_path):
            self.message.emit(None, None, 'Loading COCO json: {}'.format(self.coco_json_path))

            with open(self.coco_json_path, 'r') as f:
                dataset = load(f)
                images = {image.get('id', None): {
                    'file_name': image.get('file_name', ''),
                    'height': image.get('height', ''),
                    'width': image.get('width', ''),
                } for image in dataset.get('images', [])}
                self.message.emit(None, None, '    Contain {} images.'.format(len(images)))

                annotations = dataset.get('annotations', [])
                self.message.emit(None, None, '    Contain {} annotations.'.format(len(annotations)))

                categories = {categorie.get('id', None): {'name': categorie.get('name', '')} for categorie in
                              dataset.get('categories', [])}
                self.message.emit(None, None, '    Contain {} categories.'.format(len(categories)))

                self.message.emit(None, None, 'Loading annotations...')
                for index, annotation in enumerate(annotations):
                    if self.cancel:
                        return
                    self.message.emit(index+1, len(annotations), None)

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
                    elif iscrowd == 1 and self.keep_crowd:
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

                self.message.emit(None, None, 'Start convert to ISAT json...')
                for index, (image_id, values) in enumerate(annos.items()):
                    if self.cancel:
                        return
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
                            groups_dict[group] = len(groups_dict) + 1
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
                    save_json = os.path.join(self.to_root, json_name)

                    self.message.emit(index + 1, len(annos), '{:>8d}/{:<8d} | Converting to {}'.format(index + 1, len(annos), json_name))

                    with open(save_json, 'w') as f:
                        try:
                            dump(isat_anno, f)
                            self.message.emit(None, None, ' ' * 18 + '| Saved finished.')

                        except Exception as e:
                            self.message.emit(index + 1, len(annos), ' ' * 18 + '| Save error: {}'.format(e))

                ### 类别文件
                cmap = imgviz.label_colormap()
                sorted(categories)
                for index, (k, categorie_dict) in enumerate(categories.items()):
                    r, g, b = cmap[index + 1]
                    categorie_dict['color'] = "#{:02x}{:02x}{:02x}".format(r, g, b)

                s = yaml.dump({'label': list(categories.values())})
                with open(os.path.join(self.to_root, 'categorys.yaml'), 'w') as f:
                    f.write(s)
                    self.message.emit(None, None, 'Generate categorys.yaml.')
        else:
            self.message.emit(None, None, '{} not exist.'.format(self.coco_json_path))
        self.message.emit(None, None, '*** Finished! ***')

    def __del__(self):
        self.wait()