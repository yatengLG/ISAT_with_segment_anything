# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtWidgets, QtCore, QtGui
from ISAT.ui.file_dock import Ui_Form
import os


class FilesDockWidget(QtWidgets.QWidget, Ui_Form):
    def __init__(self, mainwindow):
        super(FilesDockWidget, self).__init__()
        self.setupUi(self)
        self.mainwindow = mainwindow
        self.listWidget.clicked.connect(self.listwidget_doubleclick)
        self.lineEdit_jump.returnPressed.connect(self.mainwindow.jump_to)

    def generate_item_and_itemwidget(self, file_name):
        item = QtWidgets.QListWidgetItem()
        item.setSizeHint(QtCore.QSize(200, 30))
        item_widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(9, 1, 9, 1)

        state_color = QtWidgets.QLabel()
        state_color.setFixedWidth(5)
        state_color.setStyleSheet("background-color: {};".format('#999999'))
        state_color.setObjectName('state_color')
        layout.addWidget(state_color)

        category = QtWidgets.QLabel(file_name)
        category.setObjectName('category')
        layout.addWidget(category)

        item_widget.setLayout(layout)
        return item, item_widget

    def update_widget(self):
        self.listWidget.clear()
        if self.mainwindow.files_list is None:
            return

        for file_path in self.mainwindow.files_list:
            _, file_name = os.path.split(file_path)
            item = QtWidgets.QListWidgetItem()
            item.setSizeHint(QtCore.QSize(200, 30))
            # item, item_widget = self.generate_item_and_itemwidget(file_name)

            item.setText(file_name)
            self.listWidget.addItem(item)
            # self.listWidget.setItemWidget(item, item_widget)

        self.label_all.setText('{}'.format(len(self.mainwindow.files_list)))

    def set_select(self, row):
        self.listWidget.setCurrentRow(row)

    def listwidget_doubleclick(self):
        row = self.listWidget.currentRow()
        self.mainwindow.current_index = row
        self.mainwindow.show_image(row)
