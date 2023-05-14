# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtWidgets, QtCore, QtGui
from ui.ISAT_to_VOC_dialog import Ui_Dialog
from tools.toVOC import TOVOC
from configs import load_config
import os


class ISATtoVOCDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent, mainwindow):
        super(ISATtoVOCDialog, self).__init__(parent)
        self.setupUi(self)
        self.mainwindow = mainwindow
        self.label_root = None
        self.save_root = None

        self.converter = TOVOC()
        self.converter.message.connect(self.print_message)

        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        self.init_connect()

    def reset_gui(self):
        self.lineEdit_label_root.clear()
        self.lineEdit_save_root.clear()
        self.checkBox_is_instance.setChecked(False)
        self.checkBox_keep_crowd.setChecked(False)
        self.textBrowser.clear()
        self.progressBar.reset()

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

    def cancel(self):
        self.converter.cancel = True
        self.close()

    def apply(self):
        if self.label_root is None or self.save_root is None:
            return
        # 语义分割，保存类别文件
        if not self.checkBox_is_instance.isChecked():
            with open(os.path.join(self.save_root, 'classesition.txt'), 'w') as f:
                for index, label in enumerate(self.mainwindow.cfg.get('label', [])):
                    f.write('{} {}\n'.format(label.get('name'), index))

        self.pushButton_label_root.setEnabled(False)
        self.pushButton_save_root.setEnabled(False)
        self.checkBox_is_instance.setEnabled(False)
        self.checkBox_keep_crowd.setEnabled(False)
        self.pushButton_apply.setEnabled(False)

        self.progressBar.reset()
        self.textBrowser.clear()
        self.converter.cancel = False
        if os.path.exists(os.path.join(self.label_root, 'isat.yaml')):
            self.converter.cfg = load_config(os.path.join(self.label_root, 'isat.yaml'))
        else:
            self.converter.cfg = self.mainwindow.cfg
        self.converter.from_root = self.label_root
        self.converter.to_root = self.save_root
        self.converter.is_instance = self.checkBox_is_instance.isChecked()
        self.converter.keep_crowd = self.checkBox_keep_crowd.isChecked()
        self.converter.start()

        self.pushButton_label_root.setEnabled(True)
        self.pushButton_save_root.setEnabled(True)
        self.checkBox_is_instance.setEnabled(True)
        self.checkBox_keep_crowd.setEnabled(True)
        self.pushButton_apply.setEnabled(True)

    def print_message(self, index, all, message):
        if all:
            self.progressBar.setMaximum(all)
        if index:
            self.progressBar.setValue(index)
        if message:
            self.textBrowser.append(message)

    def init_connect(self):
        self.pushButton_label_root.clicked.connect(self._label_root)
        self.pushButton_save_root.clicked.connect(self._save_root)
        self.pushButton_apply.clicked.connect(self.apply)
        self.pushButton_cancel.clicked.connect(self.cancel)