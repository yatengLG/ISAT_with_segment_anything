# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtWidgets

class RightButtonMenu(QtWidgets.QMenu):
    def __init__(self, mainwindow):
        super(RightButtonMenu, self).__init__()
        self.mainwindow = mainwindow

