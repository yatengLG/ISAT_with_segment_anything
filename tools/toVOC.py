# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5.QtCore import QThread, pyqtSignal
from json import load
import numpy as np
from PIL import Image, ImageColor
import mahotas
import imgviz
import os


class TOVOC(QThread):
    message = pyqtSignal(int, int, str)

    def __init__(self,):
        super(TOVOC, self).__init__()
        self.cfg = None
        self.from_root = None
        self.to_root = None
        self.is_instance = False
        self.keep_crowd = False
        self.cancel = False

    def run(self):
        labels = self.cfg.get('label', [])
        category_dict = {}
        if self.is_instance:
            cmap = imgviz.label_colormap()
        else:
            self.message.emit(None, None, 'Loading category index and color...')
            self.message.emit(None, None, '    '+'-'*53)
            self.message.emit(None, None, '    | {:^5s} | {:^8s} | {:^30s} |'.format('index', 'color', 'category'))
            cmap = np.zeros((len(labels), 3), dtype=np.uint8)
            for index, label_dict in enumerate(labels):
                category = label_dict.get('name', 'unknow')
                color = label_dict.get('color', '#000000')
                category_dict[category] = {'index': index, 'color': color}
                cmap[index] = (ImageColor.getrgb(color))
                self.message.emit(None, None, '    | {:>5d} | {:>8s} | {:<30s} |'.format(index, color, category))
            self.message.emit(None, None, '    '+'-' * 53)
            self.message.emit(None, None, 'Load category index and color finished!')

        jsons = [f for f in os.listdir(self.from_root) if f.endswith('.json')]
        num_jsons = len(jsons)

        self.message.emit(None, None, 'Start convert.')

        for index, json in enumerate(jsons):

            if self.cancel:
                return
            from_path = os.path.join(self.from_root, json)
            self.message.emit(index+1, num_jsons, '{:>8d}/{:<8d} | Loading json:{}'.format(index+1, num_jsons, json))
            with open(from_path, 'r') as f:
                dataset = load(f)
                info = dataset.get('info', {})
                objects = dataset.get('objects', [])

                img_name = info.get('name', '')
                width = info.get('width', 0)
                height = info.get('height', 0)
                depth = info.get('depth', 0)
                note = info.get('note', '')
                img = np.zeros(shape=(height, width), dtype=np.uint8)

                objects = sorted(objects, key=lambda obj:obj.get('layer', 1))

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
                    segmentation = [(int(p[1]), int(p[0])) for p in segmentation]

                    if self.is_instance and group != '':
                        mahotas.polygon.fill_polygon(segmentation, img, color=int(group))
                    else:
                        mahotas.polygon.fill_polygon(segmentation, img, color=category_dict.get(category, {}).get('index', 0))

            to_name = json[:-5]+'.png'
            to_path = os.path.join(self.to_root, to_name)
            try:
                img = Image.fromarray(img.astype(np.uint8), mode='P')

                img.putpalette(cmap.flatten())
                img.save(to_path)
                self.message.emit(None, None, ' ' * 18 + '| Saved png   :{}'.format(to_name))

            except Exception as e:
                self.message.emit(None, None, ' ' * 18 + '| Save png    :{} | error: {}'.format(to_name, e))

        self.message.emit(None, None, '*** Finished! ***')

    def __del__(self):
        self.wait()
