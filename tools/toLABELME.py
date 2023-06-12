# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5.QtCore import QThread, pyqtSignal
from json import load, dump
import numpy as np
import os

class TOLABELME(QThread):
    message = pyqtSignal(int, int, str)

    def __init__(self,):
        super(TOLABELME, self).__init__()
        self.cfg = None
        self.from_root = None
        self.to_root = None
        self.keep_crowd = False
        self.cancel = False

    def run(self):

        jsons = [f for f in os.listdir(self.from_root) if f.endswith('.json')]
        num_jsons = len(jsons)

        self.message.emit(None, None, 'Start convert.')

        for index, json in enumerate(jsons):

            if self.cancel:
                return

            from_path = os.path.join(self.from_root, json)
            self.message.emit(index+1, num_jsons, '{:>8d}/{:<8d} | Loading ISAT json:{}'.format(index+1, num_jsons, json))
            with open(from_path, 'r') as f:
                dataset = load(f)
                info = dataset.get('info', {})
                objects = dataset.get('objects', [])

                img_name = info.get('name', '')
                width = info.get('width', 0)
                height = info.get('height', 0)
                depth = info.get('depth', 0)
                note = info.get('note', '')

                objects = sorted(objects, key=lambda obj:obj.get('layer', 1))

                labelme_anno = {}
                labelme_anno['version'] = "5.2.0.post4 | ISAT to LabelMe"
                labelme_anno['imagePath'] = img_name
                labelme_anno['imageData'] = None
                labelme_anno['imageHeight'] = height
                labelme_anno['imageWidth'] = width
                labelme_anno['flags'] = {}
                labelme_anno['shapes'] = []

                for obj in objects:
                    category = obj.get('category', 'unknow')
                    group = obj.get('group', '')
                    segmentation = obj.get('segmentation', [])
                    iscrowd = obj.get('iscrowd', 0)
                    if iscrowd:
                        if not self.keep_crowd:
                            continue
                    note = obj.get('note', '')
                    area = obj.get('area', 0)
                    layer = obj.get('layer', 1)
                    bbox = obj.get('bbox', [])

                    shape = {}
                    shape['label'] = category
                    shape['points'] = segmentation
                    shape['group_id'] = int(group) if group else None
                    shape['description'] = note
                    shape['shape_type'] = 'polygon'
                    shape['flags'] = {}

                    labelme_anno['shapes'].append(shape)

                to_path = os.path.join(self.to_root, json)
                with open(to_path, 'w') as f:
                    try:
                        dump(labelme_anno, f, indent=4)
                        self.message.emit(None, None, ' ' * 18 + '| Saved labelme json: {}'.format(to_path))
                    except Exception as e:
                        self.message.emit(None, None, ' ' * 18 + '| Error: {}'.format(e))

        self.message.emit(None, None, '*** Finished! ***')

    def __del__(self):
        self.wait()
