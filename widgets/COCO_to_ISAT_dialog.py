# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtWidgets, QtCore, QtGui
from ui.COCO_to_ISAT_dialog import Ui_Dialog
from tools.toCOCO import COCOConverter


class COCOtoISATDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent, mainwindow):
        super(COCOtoISATDialog, self).__init__(parent)
        self.setupUi(self)
        self.mainwindow = mainwindow
        self.label_path = None
        self.save_root = None
        self.pause = False

        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        self.init_connect()

    def reset_gui(self):
        self.lineEdit_label_path.clear()
        self.lineEdit_save_root.clear()

    def _label_path(self):
        path, suffix = QtWidgets.QFileDialog.getOpenFileName(self, caption='COCO json save file',
                                                             filter="json (*.json)")
        if path:
            self.label_path = path
            self.lineEdit_label_path.setText(path)
        else:
            self.lineEdit_label_path.clear()

    def _save_root(self):

        dir = QtWidgets.QFileDialog.getExistingDirectory(self, caption='ISAT jsons root')
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
        if self.label_path is None or self.save_root is None:
            return

        converter = COCOConverter()

        self.pushButton_label_path.setEnabled(False)
        self.pushButton_save_root.setEnabled(False)
        self.label_info.setText('Convering...')
        converter.convert_from_coco(self.label_path, self.save_root, keep_crowd=self.checkBox_keepcrowd.isChecked())
        self.label_info.setText('Finish!!!')

        self.pushButton_label_path.setEnabled(True)
        self.pushButton_save_root.setEnabled(True)

    def init_connect(self):
        self.pushButton_label_path.clicked.connect(self._label_path)
        self.pushButton_save_root.clicked.connect(self._save_root)
        self.pushButton_apply.clicked.connect(self.apply)
        self.pushButton_cache.clicked.connect(self.cache)