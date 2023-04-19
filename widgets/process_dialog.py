# -*- coding: utf-8 -*-
# @Author  : LG

from ui.process_dialog import Ui_Dialog
from PyQt5 import QtGui, QtWidgets, QtCore

class PorcessDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent):
        super(PorcessDialog, self).__init__(parent)
        self.setupUi(self)

