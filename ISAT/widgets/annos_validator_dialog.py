# -*- coding: utf-8 -*-
# @Author: LG

from PyQt5 import QtWidgets, QtCore, QtGui
from ISAT.ui.annos_validator import Ui_Dialog
from shapely.geometry import Polygon
from shapely.validation import explain_validity
import json
import os


class AnnosValidatorDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent, mainwindow):
        super(AnnosValidatorDialog, self).__init__(parent)
        self.setupUi(self)
        self.mainwindow = mainwindow
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        self.pushButton_json_root.clicked.connect(self.open_dir)
        self.pushButton_start.clicked.connect(self.validate)
        self.pushButton_start.setEnabled(False)

    def open_dir(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(self, caption='Open dir')
        if dir:
            self.lineEdit_json_root.setText(dir)
            self.pushButton_start.setEnabled(True)
        else:
            self.lineEdit_json_root.clear()
            self.pushButton_start.setEnabled(False)

    def validate(self):
        root = self.lineEdit_json_root.text()
        json_files = [f for f in os.listdir(root) if f.endswith('.json')]

        if len(json_files) < 1:
            self.progressBar.setMaximum(1)
            self.progressBar.setValue(1)
            return

        self.progressBar.setMaximum(len(json_files))
        self.textBrowser.clear()

        for index, json_file in enumerate(json_files):
            json_file_path = os.path.join(root, json_file)
            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    info = data['info']
                    width = info['width']
                    height = info['height']

                    objects = data['objects']
                    for obj_index, obj in enumerate(objects):
                        points = obj['segmentation']
                        bbox = obj['bbox']

                        # 检查多边形顶点数量
                        if len(points) < 3:
                            self.textBrowser.append(
                                ' Error  | {} | {} polygon error. Vertex < 3.'
                                .format(json_file,
                                        '{}th'.format(
                                            obj_index) if obj_index < 20 else '{}st'.format(obj_index))
                            )

                        else:
                            # 检查多边形自交等问题
                            polyon = Polygon(points)
                            if not polyon.is_valid:
                                self.textBrowser.append(
                                    'Warning | {} | {} polygon invalid. {}'
                                    .format(json_file,
                                            '{}th'.format(
                                                obj_index) if obj_index < 20 else '{}st'.format(obj_index),
                                            explain_validity(polyon))
                                )

                        xs = [point[0] for point in points]
                        ys = [point[1] for point in points]
                        xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)

                        if xmin < 0 or xmax > width or ymin < 0 or ymax > height:
                            self.textBrowser.append(
                                'Warning | {} | {} polygon warning. {}'
                                .format(json_file,
                                        '{}th'.format(obj_index) if obj_index < 20 else '{}st'.format(obj_index),
                                        'Out of the image')
                            )
            except Exception as e:
                self.textBrowser.append(
                    " Error  | {} | Broken json file. {}".format(json_file, e)
                )
            self.progressBar.setValue(index + 1)
