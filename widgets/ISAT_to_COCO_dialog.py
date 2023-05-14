# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtWidgets, QtCore, QtGui
from ui.ISAT_to_COCO_dialog import Ui_Dialog
from tools.toCOCO import TOCOCO


class ISATtoCOCODialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent, mainwindow):
        super(ISATtoCOCODialog, self).__init__(parent)
        self.setupUi(self)
        self.mainwindow = mainwindow
        self.label_root = None
        self.save_path = None

        self.converter = TOCOCO()
        self.converter.message.connect(self.print_message)

        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.init_connect()

    def reset_gui(self):
        self.lineEdit_label_root.clear()
        self.lineEdit_save_path.clear()
        self.textBrowser.clear()
        self.progressBar.reset()

    def _label_root(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(self, caption='ISAT jsons root')
        if dir:
            self.label_root = dir
            self.lineEdit_label_root.setText(dir)
        else:
            self.lineEdit_label_root.clear()

    def _save_path(self):
        path, suffix = QtWidgets.QFileDialog.getSaveFileName(self, caption='COCO json save file', filter="json (*.json)")
        if path:
            if not path.endswith('.json'):
                path += '.json'
            self.save_path = path
            self.lineEdit_save_path.setText(path)
        else:
            self.lineEdit_save_path.clear()

    def cancel(self):
        self.converter.cancel = True
        self.close()

    def apply(self):
        if self.label_root is None or self.save_path is None:
            return

        self.pushButton_label_root.setEnabled(False)
        self.pushButton_save_path.setEnabled(False)
        self.pushButton_apply.setEnabled(False)

        self.progressBar.reset()
        self.textBrowser.clear()
        self.converter.cancel = False
        self.converter.isat_json_root = self.label_root
        self.converter.to_path = self.save_path
        self.converter.start()

        self.pushButton_label_root.setEnabled(True)
        self.pushButton_save_path.setEnabled(True)
        self.pushButton_apply.setEnabled(True)

    def print_message(self, index, all, message):
        if index:
            self.progressBar.setValue(index)
        if all:
            self.progressBar.setMaximum(all)
        if message:
            self.textBrowser.append(message)

    def init_connect(self):
        self.pushButton_label_root.clicked.connect(self._label_root)
        self.pushButton_save_path.clicked.connect(self._save_path)
        self.pushButton_apply.clicked.connect(self.apply)
        self.pushButton_cancel.clicked.connect(self.cancel)