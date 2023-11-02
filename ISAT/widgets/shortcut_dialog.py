# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtGui, QtCore, QtWidgets
from ISAT.ui.shortcut_dialog import Ui_Dialog

class ShortcutDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent):
        super(ShortcutDialog, self).__init__(parent)
        self.setupUi(self)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)