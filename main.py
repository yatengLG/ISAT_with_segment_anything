# -*- coding: utf-8 -*-
# @Author  : LG
# import os

from PyQt5 import QtWidgets
from widgets.mainwindow import MainWindow
import sys

# ubuntu18.04装其他应用时，环境崩了。。。qtapp初始化后，torch调用cuda会卡死，但先调用下cuda就不会卡死。
import torch
torch.cuda.is_available()
# (知道怎么处理该问题，请反馈下 T.T)


if __name__ == '__main__':

    app = QtWidgets.QApplication([''])
    mainwindow = MainWindow()
    mainwindow.show()
    sys.exit(app.exec_())

