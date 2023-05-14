# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtWidgets, QtCore
from ui.COCO_to_ISAT_dialog import Ui_Dialog
from tools.fromCOCO import FROMCOCO


class COCOtoISATDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent, mainwindow):
        super(COCOtoISATDialog, self).__init__(parent)
        self.setupUi(self)
        self.mainwindow = mainwindow
        self.label_path = None
        self.save_root = None

        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        self.converter = FROMCOCO()
        self.converter.message.connect(self.print_message)

        self.init_connect()

    def reset_gui(self):
        self.lineEdit_label_path.clear()
        self.lineEdit_save_root.clear()
        self.checkBox_keepcrowd.setChecked(False)
        self.progressBar.reset()
        self.textBrowser.clear()

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

    def cancel(self):
        self.converter.cancel = True
        self.close()

    def apply(self):
        if self.label_path is None or self.save_root is None:
            return

        self.pushButton_label_path.setEnabled(False)
        self.pushButton_save_root.setEnabled(False)
        self.checkBox_keepcrowd.setEnabled(False)
        self.pushButton_apply.setEnabled((False))

        self.progressBar.reset()
        self.textBrowser.clear()
        self.converter.cancel = False
        self.converter.coco_json_path = self.label_path
        self.converter.to_root = self.save_root
        self.converter.keep_crowd = self.checkBox_keepcrowd.isChecked()
        self.converter.run()

        self.pushButton_label_path.setEnabled(True)
        self.pushButton_save_root.setEnabled(True)
        self.checkBox_keepcrowd.setEnabled(True)
        self.pushButton_apply.setEnabled((True))

    def print_message(self, index, all, message):
        if all:
            self.progressBar.setMaximum(all)
        if index:
            self.progressBar.setValue(index)
        if message:
            self.textBrowser.append(message)

    def init_connect(self):
        self.pushButton_label_path.clicked.connect(self._label_path)
        self.pushButton_save_root.clicked.connect(self._save_root)
        self.pushButton_apply.clicked.connect(self.apply)
        self.pushButton_cancel.clicked.connect(self.cancel)