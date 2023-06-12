# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtWidgets, QtCore, QtGui
from ui.ISAT_to_LABELME_dialog import Ui_Dialog
from tools.toLABELME import TOLABELME
import os

class ISATtoLabelMeDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent, mainwindow):
        super(ISATtoLabelMeDialog, self).__init__(parent)
        self.setupUi(self)
        self.mainwindow = mainwindow
        self.label_root = None
        self.save_root = None

        self.converter = TOLABELME()
        self.converter.message.connect(self.print_message)

        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        self.init_connect()

    def reset_gui(self):
        self.lineEdit_fromroot.clear()
        self.lineEdit_toroot.clear()
        self.checkBox_keepcrowd.setChecked(False)
        self.textBrowser.clear()
        self.progressBar.reset()


    def _label_root(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(self)
        if dir:
            self.label_root = dir
            self.lineEdit_fromroot.setText(dir)
        else:
            self.lineEdit_fromroot.clear()

    def _save_root(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(self)
        if dir:
            self.save_root = dir
            self.lineEdit_toroot.setText(dir)
        else:
            self.lineEdit_toroot.clear()

    def cancel(self):
        self.converter.cancel = True
        self.close()

    def apply(self):
        if self.label_root is None or self.save_root is None:
            return

        self.pushButton_fromroot.setEnabled(False)
        self.pushButton_toroot.setEnabled(False)
        self.checkBox_keepcrowd.setEnabled(False)
        self.pushButton_apply.setEnabled(False)
        self.progressBar.reset()
        self.textBrowser.clear()
        self.converter.cancel = False

        self.converter.from_root = self.label_root
        self.converter.to_root = self.save_root
        self.converter.keep_crowd = self.checkBox_keepcrowd.isChecked()
        self.converter.start()

        self.pushButton_fromroot.setEnabled(True)
        self.pushButton_toroot.setEnabled(True)
        self.checkBox_keepcrowd.setEnabled(True)
        self.pushButton_apply.setEnabled(True)

    def print_message(self, index, all, message):
        if all:
            self.progressBar.setMaximum(all)
        if index:
            self.progressBar.setValue(index)
        if message:
            self.textBrowser.append(message)

    def init_connect(self):
        self.pushButton_fromroot.clicked.connect(self._label_root)
        self.pushButton_toroot.clicked.connect(self._save_root)
        self.pushButton_apply.clicked.connect(self.apply)
        self.pushButton_cancel.clicked.connect(self.cancel)