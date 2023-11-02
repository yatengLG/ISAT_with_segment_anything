# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtWidgets, QtGui, QtCore
from ISAT.ui.category_choice import Ui_Dialog
from ISAT.configs import load_config, CONFIG_FILE, DEFAULT_CONFIG_FILE
import os


class CategoryChoiceDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent, mainwindow, scene):
        super(CategoryChoiceDialog, self).__init__(parent)

        self.setupUi(self)
        self.mainwindow = mainwindow
        self.scene = scene

        self.lineEdit_group.setValidator(QtGui.QIntValidator(0, 1000))

        self.listWidget.itemClicked.connect(self.get_category)
        self.pushButton_apply.clicked.connect(self.apply)
        self.pushButton_cancel.clicked.connect(self.cancel)

        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

    def load_cfg(self):
        self.listWidget.clear()

        labels = self.mainwindow.cfg.get('label', [])

        for label in labels:
            name = label.get('name', 'UNKNOW')
            color = label.get('color', '#000000')
            item = QtWidgets.QListWidgetItem()
            item.setSizeHint(QtCore.QSize(200, 30))
            widget = QtWidgets.QWidget()

            layout = QtWidgets.QHBoxLayout()
            layout.setContentsMargins(9, 1, 9, 1)
            label_category = QtWidgets.QLabel()
            label_category.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            label_category.setText(name)
            label_category.setObjectName('label_category')

            label_color = QtWidgets.QLabel()
            label_color.setFixedWidth(10)
            label_color.setStyleSheet("background-color: {};".format(color))
            label_color.setObjectName('label_color')

            layout.addWidget(label_color)
            layout.addWidget(label_category)
            widget.setLayout(layout)

            self.listWidget.addItem(item)
            self.listWidget.setItemWidget(item, widget)

        self.lineEdit_group.clear()
        self.lineEdit_category.clear()
        self.checkBox_iscrowded.setCheckState(False)
        self.label_layer.setText('{}'.format(len(self.mainwindow.polygons)+1))

        if self.listWidget.count() == 0:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Please set categorys before tagging.')

    def get_category(self, item):
        widget = self.listWidget.itemWidget(item)
        label_category = widget.findChild(QtWidgets.QLabel, 'label_category')
        self.lineEdit_category.setText(label_category.text())
        self.lineEdit_category.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

    def apply(self):
        category = self.lineEdit_category.text()
        group = int(self.lineEdit_group.text())
        is_crowd = int(self.checkBox_iscrowded.isChecked())
        note = self.lineEdit_note.text()
        if not category:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Please select one category before submitting.')
            return

        # 设置polygon 属性
        self.scene.current_graph.set_drawed(category, group, is_crowd, note,
                                            QtGui.QColor(self.mainwindow.category_color_dict[category]), self.scene.top_layer)
        # 添加新polygon
        self.mainwindow.polygons.append(self.scene.current_graph)
        # 设置为最高图层
        self.scene.current_graph.setZValue(len(self.mainwindow.polygons))
        for vertex in self.scene.current_graph.vertexs:
            vertex.setZValue(len(self.mainwindow.polygons))

        self.mainwindow.annos_dock_widget.update_listwidget()

        self.scene.current_graph = None
        self.scene.change_mode_to_view()
        self.close()

    def cancel(self):
        self.scene.cancel_draw()
        self.close()

    def closeEvent(self, a0: QtGui.QCloseEvent):
        self.cancel()

    def reject(self):
        self.cancel()
