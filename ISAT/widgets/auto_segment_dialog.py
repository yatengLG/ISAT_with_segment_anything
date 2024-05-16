# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QThread, pyqtSignal
from ISAT.ui.auto_segment import Ui_Dialog
from ISAT.configs import CONTOURMode
from xml.etree import ElementTree as ET
from PIL import Image
import numpy as np
from json import dump
import imgviz
import yaml
import cv2
import os

class AutoSegmentThread(QThread):
    message = pyqtSignal(int, int, str)
    def __init__(self, mainwindow):
        super(AutoSegmentThread, self).__init__()
        self.mainwindow = mainwindow
        self.cancel = False
        self.image_dir = None
        self.label_dir = None
        self.save_dir = None

    def run(self):
        image_names = []
        cates = set()
        suffixs = tuple(
            ['{}'.format(fmt.data().decode('ascii').lower()) for fmt in QtGui.QImageReader.supportedImageFormats()])
        for f in os.listdir(self.image_dir):
            if f.lower().endswith(suffixs):
                image_names.append(f)
        image_names.sort()
        images_num = len(image_names)

        for index, image_name in enumerate(image_names):
            self.message.emit(index+1, images_num, '{}'.format(image_name))
            image_path = os.path.join(self.image_dir, image_name)
            if self.cancel:
                self.message.emit(-1, -1, '{}'.format('Cancel!'))
                return
            xml_name = '.'.join(image_name.split('.')[:-1]) + '.xml'
            xml_path = os.path.join(self.label_dir, xml_name)
            if not os.path.exists(xml_path):
                self.message.emit(-1, -1, '{}'.format("Don't exist xml file."))
                continue
            self.message.emit(-1, -1, '{}'.format('Sam encoding...'))
            # sam
            try:
                image_data = np.array(Image.open(image_path))
                self.mainwindow.segany.set_image(image_data)
            except Exception as e:
                self.message.emit(-1, -1, 'Sam error when encoding image: {}'.format(e))
                continue

            # xml
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                objs = root.findall('object')
                size = root.find('size')
                width = size.find('width').text
                height = size.find('height').text
                depth = size.find('depth').text
            except Exception as e:
                self.message.emit(-1, -1, 'Load xml error: {}'.format(e))
                continue

            # isat
            dataset = {}
            dataset['info'] = {}
            dataset['info']['description'] = 'ISAT'
            dataset['info']['folder'] = self.image_dir
            dataset['info']['name'] = image_name
            dataset['info']['width'] = width
            dataset['info']['height'] = height
            dataset['info']['depth'] = depth
            dataset['info']['note'] = ''
            dataset['objects'] = []

            for group, obj in enumerate(objs):
                name = obj.find('name').text
                difficult = obj.find('difficult').text
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                cates.add(name)

                masks = self.mainwindow.segany.predict_with_box_prompt(box=np.array([xmin, ymin, xmax, ymax]))

                masks = masks.astype('uint8') * 255
                h, w = masks.shape[-2:]
                masks = masks.reshape(h, w)

                if self.mainwindow.scene.contour_mode == CONTOURMode.SAVE_ALL:
                    # 当保留所有轮廓时，检测所有轮廓，并建立二层等级关系
                    contours, hierarchy = cv2.findContours(masks, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
                else:
                    # 当只保留外轮廓或单个mask时，只检测外轮廓
                    contours, hierarchy = cv2.findContours(masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

                if self.mainwindow.scene.contour_mode == CONTOURMode.SAVE_MAX_ONLY:
                    largest_contour = max(contours, key=cv2.contourArea)  # 只保留面积最大的轮廓
                    contours = [largest_contour]

                for _, contour in enumerate(contours):
                    # polydp
                    if self.mainwindow.cfg['software']['use_polydp']:
                        epsilon_factor = 0.001
                        epsilon = epsilon_factor * cv2.arcLength(contour, True)
                        contour = cv2.approxPolyDP(contour, epsilon, True)

                    object = {}
                    object['category'] = name
                    object['group'] = group + 1
                    object['segmentation'] = [(int(point[0][0]), int(point[0][1])) for point in contour]
                    object['area'] = None
                    object['layer'] = group + 1
                    object['bbox'] = [xmin, ymin, xmax, ymax]
                    object['iscrowd'] = 0
                    object['note'] = ''
                    dataset['objects'].append(object)

            try:
                save_path = os.path.join(self.save_dir, '.'.join(image_name.split('.')[:-1]) + '.json')
                with open(save_path, 'w') as f:
                    dump(dataset, f, indent=4)
                self.message.emit(-1, -1, '{}'.format('Save finished!'))
            except Exception as e:
                self.message.emit(-1, -1, 'Save ISAT json error: {}'.format(e))

        # 类别文件
        try:
            cmap = imgviz.label_colormap()
            cates = list(cates)
            cates = sorted(cates)
            categories = []
            for index, cat in enumerate(cates):
                r, g, b = cmap[index + 1]
                categories.append({
                    'name': cat if isinstance(cat, str) else str(cat),
                    'color': "#{:02x}{:02x}{:02x}".format(r, g, b)
                })
            s = yaml.dump({'label': categories})
            with open(os.path.join(self.save_dir, 'isat.yaml'), 'w') as f:
                f.write(s)
            self.message.emit(-1, -1, 'Save ISAT yaml finished!')

        except Exception as e:
            self.message.emit(-1, -1, 'Save ISAT yaml error: {}'.format(e))


class AutoSegmentDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent, mainwindow):
        super(AutoSegmentDialog, self).__init__(parent)
        self.setupUi(self)
        self.mainwindow = mainwindow
        self.image_dir = None
        self.label_dir = None
        self.save_dir = None

        self.auto_segment_thread = AutoSegmentThread(self.mainwindow)
        self.auto_segment_thread.message.connect(self.print_message)

        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.init_connect()

    def open_dir(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(self, caption='Open dir')
        if self.sender() == self.pushButton_image_dir:
            lineEdit = self.lineEdit_image_dir
        elif self.sender() == self.pushButton_label_dir:
            lineEdit = self.lineEdit_label_dir
        elif self.sender() == self.pushButton_save_dir:
            lineEdit = self.lineEdit_save_dir
        else:
            return
        if dir:
            if lineEdit is not None:
                lineEdit.setText(dir)
        else:
            if lineEdit is not None:
                lineEdit.clear()

    def start(self):
        self.auto_segment_thread.cancel = False

        self.image_dir = self.lineEdit_image_dir.text()
        self.label_dir = self.lineEdit_label_dir.text()
        self.save_dir = self.lineEdit_save_dir.text()

        if self.image_dir == '' or self.label_dir == '' or self.save_dir == '':
            self.textBrowser.append('不能为空')
            return

        self.auto_segment_thread.image_dir = self.image_dir
        self.auto_segment_thread.label_dir = self.label_dir
        self.auto_segment_thread.save_dir = self.save_dir
        self.auto_segment_thread.start()

    def cancel(self):
        self.auto_segment_thread.cancel = True

    def print_message(self, index, all, message):
        if index > 0:
            self.progressBar.setValue(index)
        if all > 0:
            self.progressBar.setMaximum(all)
        if message:
            self.textBrowser.append('{} | {}'.format(
                '{:>8s}/{:<8s}'.format(str(index), str(all)) if (index > 0 and all > 0) else '{:>8s} {:<8s}'.format('', ''), message))
            print('{} | {}'.format(
                '{:>8s}/{:<8s}'.format(str(index), str(all)) if (index > 0 and all > 0) else '{:>8s} {:<8s}'.format('', ''), message))

    def init_connect(self):
        self.pushButton_image_dir.clicked.connect(self.open_dir)
        self.pushButton_label_dir.clicked.connect(self.open_dir)
        self.pushButton_save_dir.clicked.connect(self.open_dir)
        self.pushButton_start.clicked.connect(self.start)
        self.pushButton_cancel.clicked.connect(self.cancel)
