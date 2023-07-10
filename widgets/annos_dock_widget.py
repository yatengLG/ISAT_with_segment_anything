# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtWidgets, QtCore, QtGui
from ui.anno_dock import Ui_Form
import functools


class AnnosDockWidget(QtWidgets.QWidget, Ui_Form):
    def __init__(self, mainwindow):
        super(AnnosDockWidget, self).__init__()
        self.setupUi(self)
        self.mainwindow = mainwindow
        self.polygon_item_dict = {}

        self.listWidget.itemSelectionChanged.connect(self.set_polygon_selected)
        self.checkBox_visible.stateChanged.connect(self.set_all_polygon_visible)

        self.listWidget.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.listWidget.customContextMenuRequested.connect(
            self.right_button_menu)

    def right_button_menu(self, point):
        self.mainwindow.right_button_menu.exec_(self.listWidget.mapToGlobal(point))

    def generate_item_and_itemwidget(self, polygon):
        color = self.mainwindow.category_color_dict.get(polygon.category, '#000000')
        item = QtWidgets.QListWidgetItem()
        item.setSizeHint(QtCore.QSize(200, 30))
        item_widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(9, 1, 9, 1)
        check_box = QtWidgets.QCheckBox()
        check_box.setFixedWidth(20)
        check_box.setChecked(polygon.isVisible())
        check_box.setObjectName('check_box')
        check_box.stateChanged.connect(functools.partial(self.set_polygon_show, polygon))
        layout.addWidget(check_box)

        label_color = QtWidgets.QLabel()
        label_color.setFixedWidth(10)
        label_color.setStyleSheet("background-color: {};".format(color))
        layout.addWidget(label_color)

        category = QtWidgets.QLabel(polygon.category)

        group = QtWidgets.QLabel('{}'.format(polygon.group))

        note = QtWidgets.QLabel('{}'.format(polygon.note))

        label_iscrowd = QtWidgets.QLabel()
        label_iscrowd.setFixedWidth(3)
        if polygon.iscrowd == 1:
            label_iscrowd.setStyleSheet("background-color: {};".format('#000000'))

        layout.addWidget(category)
        layout.addWidget(group)
        layout.addWidget(note)
        layout.addWidget(label_iscrowd, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

        item_widget.setLayout(layout)
        return item, item_widget

    def update_listwidget(self):
        self.listWidget.clear()
        self.polygon_item_dict.clear()
        self.checkBox_visible.setChecked(True)

        for polygon in self.mainwindow.polygons:
            item, item_widget = self.generate_item_and_itemwidget(polygon)
            self.listWidget.addItem(item)
            self.listWidget.setItemWidget(item, item_widget)
            self.polygon_item_dict[polygon] = item

        if self.mainwindow.load_finished:
            self.mainwindow.set_saved_state(False)

    def set_selected(self, polygon):
        item = self.polygon_item_dict[polygon]
        if polygon.isSelected():
            if not item.isSelected():
                item.setSelected(True)
        if not polygon.isSelected():
            if item.isSelected():
                item.setSelected(False)

    def set_polygon_selected(self):
        items = self.listWidget.selectedItems()

        have_selected = True if items else False
        if have_selected:
            self.mainwindow.scene.change_mode_to_edit()
        else:
            self.mainwindow.scene.change_mode_to_view()

        for index, polygon in enumerate(self.mainwindow.polygons):
            if self.polygon_item_dict[polygon] in items:
                if not polygon.isSelected():
                    polygon.setSelected(True)
            else:
                if polygon.isSelected():
                    polygon.setSelected(False)

    def set_polygon_show(self, polygon):
        for vertex in polygon.vertexs:
            vertex.setVisible(self.sender().checkState())
        polygon.setVisible(self.sender().checkState())

    def set_all_polygon_visible(self, visible:bool=None):
        visible = self.checkBox_visible.isChecked() if visible is None else visible
        for index in range(self.listWidget.count()):
            item = self.listWidget.item(index)
            widget = self.listWidget.itemWidget(item)
            check_box = widget.findChild(QtWidgets.QCheckBox, 'check_box')
            check_box.setChecked(visible)
        self.checkBox_visible.setChecked(visible)
