# -*- coding: utf-8 -*-
# @Author  : LG
# import os

from PyQt5 import QtWidgets
from widgets.mainwindow import MainWindow
import sys


if __name__ == '__main__':

    app = QtWidgets.QApplication([''])
    mainwindow = MainWindow()
    mainwindow.show()
    sys.exit(app.exec_())

