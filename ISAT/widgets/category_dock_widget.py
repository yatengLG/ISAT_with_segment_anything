# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtWidgets, QtCore
from ISAT.ui.category_dock import Ui_Form


class CategoriesDockWidget(QtWidgets.QWidget, Ui_Form):
    def __init__(self, mainwindow):
        super(CategoriesDockWidget, self).__init__()
        self.setupUi(self)
        self.mainwindow = mainwindow
        self.listWidget.itemClicked.connect(self.item_choice)

        # 新增 手动/自动 group 选择
        self.lineEdit_currentGroup.setText(str(self.mainwindow.current_group))
        self.lineEdit_currentGroup.textChanged.connect(self.update_current_group)
        self.pushButton_increase.clicked.connect(self.increase_current_group)
        self.pushButton_decrease.clicked.connect(self.decrease_current_group)
        self.pushButton_group_mode.clicked.connect(self.toggle_group_mode)

    def update_widget(self):
        self.listWidget.clear()
        btngroup = QtWidgets.QButtonGroup(self)
        labels = self.mainwindow.cfg.get('label', [])
        for index in range(len(labels)):
            label = labels[index]
            name = label.get('name', 'UNKNOW')
            color = label.get('color', '#000000')
            item = QtWidgets.QListWidgetItem()
            item.setSizeHint(QtCore.QSize(200, 30))
            widget = QtWidgets.QWidget()

            layout = QtWidgets.QHBoxLayout()
            layout.setContentsMargins(9, 1, 9, 1)

            label_color = QtWidgets.QLabel()
            label_color.setFixedWidth(10)
            label_color.setStyleSheet("background-color: {};".format(color))
            label_color.setObjectName('label_color')

            label_radio = QtWidgets.QRadioButton('{}'.format(name))
            label_radio.setObjectName('label_radio')
            label_radio.toggled.connect(self.radio_choice)
            btngroup.addButton(label_radio)
            if name == '__background__':
                label_radio.setChecked(True)

            layout.addWidget(label_color)
            layout.addWidget(label_radio)
            widget.setLayout(layout)

            self.listWidget.addItem(item)
            self.listWidget.setItemWidget(item, widget)

    def radio_choice(self):
        if isinstance(self.sender(), QtWidgets.QRadioButton):
            if self.sender().isChecked():
                self.mainwindow.current_category = self.sender().text()

    def item_choice(self, item_now):
        for index in range(self.listWidget.count()):
            item = self.listWidget.item(index)
            widget = self.listWidget.itemWidget(item)
            label_radio = widget.findChild(QtWidgets.QRadioButton, 'label_radio')
            label_radio.setChecked(item==item_now)

    def update_current_group(self, text):
        # Update the current_group variable when the text in the QLineEdit changes
        try:
            self.mainwindow.current_group = int(text)
        except ValueError:
            pass

    def increase_current_group(self):
        # Increase the current_group variable and update the QLineEdit text
        self.mainwindow.current_group += 1
        self.lineEdit_currentGroup.setText(str(self.mainwindow.current_group))

    def decrease_current_group(self):
        # Decrease the current_group variable and update the QLineEdit text
        if self.mainwindow.current_group > 1:
            self.mainwindow.current_group -= 1
            self.lineEdit_currentGroup.setText(str(self.mainwindow.current_group))

    def toggle_group_mode(self):
        if self.mainwindow.group_select_mode == 'auto':
            self.mainwindow.group_select_mode = 'manual'
            self.pushButton_group_mode.setText("Manual")
            self.pushButton_group_mode.setStatusTip("Manual set group id.")
        elif self.mainwindow.group_select_mode == 'manual':
            self.mainwindow.group_select_mode = 'track'
            self.pushButton_group_mode.setText("Track")
            self.pushButton_group_mode.setStatusTip("Group id changed with the group of current polygons when use [TAB] or [`] to check.")
        elif self.mainwindow.group_select_mode == 'track':
            self.mainwindow.group_select_mode = 'auto'
            self.pushButton_group_mode.setText("Auto")
            self.pushButton_group_mode.setStatusTip("Group id auto add 1 when add a new polygon.")
