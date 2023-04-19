# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtWidgets, QtCore, QtGui
from ui.file_dock import Ui_Form
import os


class FilesDockWidget(QtWidgets.QWidget, Ui_Form):
    def __init__(self, mainwindow):
        super(FilesDockWidget, self).__init__()
        self.setupUi(self)
        self.mainwindow = mainwindow
        self.listWidget.clicked.connect(self.listwidget_doubleclick)
        self.lineEdit_jump.returnPressed.connect(self.mainwindow.jump_to)

    def update_widget(self):
        self.listWidget.clear()
        if self.mainwindow.files_list is None:
            return

        for file_path in self.mainwindow.files_list:
            _, file_name = os.path.split(file_path)
            item = QtWidgets.QListWidgetItem()
            item.setSizeHint(QtCore.QSize(200, 30))

            item.setText(file_name)
            self.listWidget.addItem(item)

        self.label_all.setText('{}'.format(len(self.mainwindow.files_list)))

    def set_select(self, row):
        self.listWidget.setCurrentRow(row)

    def listwidget_doubleclick(self):
        row = self.listWidget.currentRow()
        self.mainwindow.show_image(row)
