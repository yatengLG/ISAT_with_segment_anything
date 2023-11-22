# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtWidgets, QtCore, QtGui
from ISAT.ui.setting_dialog import Ui_Dialog
from ISAT.configs import *
import os


class SettingDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent, mainwindow):
        super(SettingDialog, self).__init__(parent)
        self.mainwindow = mainwindow
        self.setupUi(self)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        self.init_connect()

    def get_item_and_widget(self, category, color: str):
        item = QtWidgets.QListWidgetItem()
        item.setSizeHint(QtCore.QSize(200, 40))

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        category_label = QtWidgets.QLabel()
        category_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        category_label.setText(category)
        category_label.setObjectName('category')
        # 颜色
        color_button = QtWidgets.QPushButton()
        color_button.setStyleSheet('QWidget {background-color: %s}' % color)
        color_button.setFixedWidth(50)
        color_button.clicked.connect(self.edit_category_item_color)
        color_button.setObjectName('color')
        # 删除
        delete_button = QtWidgets.QPushButton()
        delete_button.setText('delete')
        delete_button.setFixedWidth(80)
        delete_button.clicked.connect(self.remove_category_item)

        if category == '__background__':
            color_button.setEnabled(False)
            delete_button.setEnabled(False)

        layout.addWidget(category_label)
        layout.addWidget(color_button)
        layout.addWidget(delete_button)
        widget.setLayout(layout)
        return item, widget

    def edit_category_item_color(self):
        button = self.sender()
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            button.setStyleSheet('QWidget {background-color: %s}' % (color.name()))

    def remove_category_item(self):
        button = self.sender()
        row = self.category_list_widget.indexAt(button.parent().pos()).row()
        self.category_list_widget.takeItem(row)

    def load_cfg(self):
        self.label_config_file.setText(self.mainwindow.config_file)
        self.category_list_widget.clear()

        for name, color in self.mainwindow.category_color_dict.items():
            item, item_widget = self.get_item_and_widget(name, color=color)
            self.category_list_widget.addItem(item)
            self.category_list_widget.setItemWidget(item, item_widget)

    def add_new_category(self):
        category = self.category_input.text()
        color = self.color_button.palette().button().color().name()
        if category:
            item, item_widget = self.get_item_and_widget(category, color)
            self.category_list_widget.addItem(item)
            self.category_list_widget.setItemWidget(item, item_widget)
        self.category_input.clear()

    def choice_color(self):
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            self.color_button.setStyleSheet('QWidget {background-color: %s}' % color.name())

    def import_cfg(self):
        file, _ = QtWidgets.QFileDialog.getOpenFileName(self, filter='Yaml File(*.yaml)')
        if file:
            self.mainwindow.config_file = file
            self.mainwindow.actionSetting.setStatusTip("Config yaml: {}".format(file))
            self.mainwindow.reload_cfg()
        self.load_cfg()

    def export_cfg(self):
        file, _ = QtWidgets.QFileDialog.getSaveFileName(self, filter='Yaml File(*.yaml)')
        if not file.endswith('.yaml'):
            file += '.yaml'
        if file:
            self.mainwindow.save_cfg(file)
        self.load_cfg()

    def apply(self):
        cfg = load_config(self.mainwindow.config_file)
        cfg['label'] = []
        for index in range(self.category_list_widget.count()):
            item = self.category_list_widget.item(index)
            widget = self.category_list_widget.itemWidget(item)
            category_label = widget.findChild(QtWidgets.QLabel, 'category')
            color_button = widget.findChild(QtWidgets.QPushButton, 'color')
            cfg['label'].append(
                {'name': category_label.text(), 'color': color_button.palette().button().color().name()})

        save_config(cfg, self.mainwindow.config_file)
        self.mainwindow.reload_cfg()
        self.close()

    def cancel(self):
        self.close()

    def init_connect(self):
        self.add_button.clicked.connect(self.add_new_category)
        self.apply_button.clicked.connect(self.apply)
        self.cancel_button.clicked.connect(self.cancel)
        self.color_button.clicked.connect(self.choice_color)
        self.pushButton_import.clicked.connect(self.import_cfg)
        self.pushButton_export.clicked.connect(self.export_cfg)
