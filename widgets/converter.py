# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtWidgets, QtCore, QtGui
from ui.convert_dialog import Ui_Dialog
from tools.label_convert import Converter
import os


class ConvertDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent, mainwindow):
        super(ConvertDialog, self).__init__(parent)
        self.setupUi(self)
        self.mainwindow = mainwindow
        self.label_root = None
        self.save_root = None
        self.pause = False

        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        self.init_connect()

    def reset_gui(self):
        self.widget_process.setVisible(False)
        self.lineEdit_label_root.clear()
        self.lineEdit_save_root.clear()

    def _label_root(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(self)
        if dir:
            self.label_root = dir
            self.lineEdit_label_root.setText(dir)
        else:
            self.lineEdit_label_root.clear()

    def _save_root(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(self)
        if dir:
            self.save_root = dir
            self.lineEdit_save_root.setText(dir)
        else:
            self.lineEdit_save_root.clear()

    def cache(self):
        self.pause = True
        self.close()

    def apply(self):
        self.pause = False
        if self.label_root is None or self.save_root is None:
            return
        # 语义分割，保存类别文件
        if not self.checkBox_is_instance.isChecked():
            with open(os.path.join(self.save_root, 'classesition.txt'), 'w') as f:
                for index, label in enumerate(self.mainwindow.cfg.get('label', [])):
                    f.write('{} {}\n'.format(label.get('name'), index))

        converter = Converter(self.mainwindow.cfg, self.checkBox_is_instance.isChecked())
        jsons = [f for f in os.listdir(self.label_root) if f.endswith('.json')]

        self.pushButton_label_root.setEnabled(False)
        self.pushButton_save_root.setEnabled(False)
        self.checkBox_is_instance.setEnabled(False)

        self.widget_process.setVisible(True)
        self.progressBar.setMaximum(len(jsons))
        self.all_num.setText('{}'.format(len(jsons)))

        for index, json in enumerate(jsons):
            if self.pause:
                break
            label_path = os.path.join(self.label_root, json)
            save_path = os.path.join(self.save_root, json[:-5]+'.png')
            converter.convert(label_path, save_path)
            self.progressBar.setValue(index+1)
            self.current_num.setText('{}'.format(index+1))

        self.pushButton_label_root.setEnabled(True)
        self.pushButton_save_root.setEnabled(True)
        self.checkBox_is_instance.setEnabled(True)

    def init_connect(self):
        self.pushButton_label_root.clicked.connect(self._label_root)
        self.pushButton_save_root.clicked.connect(self._save_root)
        self.pushButton_apply.clicked.connect(self.apply)
        self.pushButton_cache.clicked.connect(self.cache)