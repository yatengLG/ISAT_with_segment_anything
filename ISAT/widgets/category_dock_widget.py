# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtWidgets, QtCore, QtGui
from ISAT.ui.category_dock import Ui_Form
from fuzzywuzzy import process
import functools


class CategoriesDockWidget(QtWidgets.QWidget, Ui_Form):
    def __init__(self, mainwindow):
        super(CategoriesDockWidget, self).__init__()
        self.setupUi(self)
        self.mainwindow = mainwindow
        self.pushButton_category_setting.clicked.connect(
            self.mainwindow.category_setting
        )
        self.listWidget.itemClicked.connect(self.item_choice)
        self.listWidget.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.lineEdit_search_category.textChanged.connect(self.update_widget)
        self.lineEdit_search_category.setClearButtonEnabled(True)

        # 新增 手动/自动 group 选择
        self.lineEdit_currentGroup.setText(str(self.mainwindow.current_group))
        self.lineEdit_currentGroup.textChanged.connect(self.update_current_group)
        self.pushButton_increase.clicked.connect(self.increase_current_group)
        self.pushButton_decrease.clicked.connect(self.decrease_current_group)
        self.pushButton_group_mode.clicked.connect(self.toggle_group_mode)

        self.category_choice_shortcuts = {}
        for i in range(10):
            shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("{}".format(i)), self)
            shortcut.activated.connect(self.choice_category)
            self.category_choice_shortcuts[shortcut] = i - 1 if i != 0 else 9

    def choice_category(self):
        """Shortcut function for category choice."""
        index = self.category_choice_shortcuts.get(self.sender(), 0)
        try:
            item = self.listWidget.item(index)
            widget = self.listWidget.itemWidget(item)
            label_radio = widget.findChild(QtWidgets.QRadioButton, "label_radio")
            label_radio.setChecked(True)
        except:
            pass

    def update_widget(self):
        """Update list widget."""
        self.listWidget.clear()
        btngroup = QtWidgets.QButtonGroup(self)
        labels = self.mainwindow.cfg.get("label", [])
        search_text = self.lineEdit_search_category.text()

        name_label_dict = {label.get("name", "UNKNOW"): label for label in labels}

        label_names = [label.get("name", "UNKNOW") for label in labels]
        if search_text == "":
            show_label_names = label_names
        elif search_text.strip(" ") == "":
            show_label_names = label_names
        else:
            matches = process.extract(search_text, label_names, limit=5)
            show_label_names = [name for name, score in matches if score > 0]

        for index in range(len(show_label_names)):
            label = name_label_dict[show_label_names[index]]
            name = label.get("name", "UNKNOW")
            color = label.get("color", "#000000")
            item = QtWidgets.QListWidgetItem()
            item.setSizeHint(QtCore.QSize(200, 30))
            widget = QtWidgets.QWidget()

            layout = QtWidgets.QHBoxLayout()
            layout.setContentsMargins(9, 1, 9, 1)

            label_color = QtWidgets.QLabel()
            label_color.setFixedWidth(10)
            label_color.setStyleSheet("background-color: {};".format(color))
            label_color.setObjectName("label_color")

            label_radio = QtWidgets.QRadioButton("{}".format(name))
            label_radio.setObjectName("label_radio")
            label_radio.toggled.connect(functools.partial(self.radio_choice, index))
            label_radio.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            btngroup.addButton(label_radio)
            if name == "__background__":
                label_radio.setChecked(True)

            layout.addWidget(label_color)
            layout.addWidget(label_radio)
            widget.setLayout(layout)

            self.listWidget.addItem(item)
            self.listWidget.setItemWidget(item, widget)

    def radio_choice(self, index: int):
        """
        Triggered when radio button selected and set current category.

        Arguments:
            index: Index of list widget item.
        """
        if isinstance(self.sender(), QtWidgets.QRadioButton):
            if self.sender().isChecked():
                self.mainwindow.current_category = self.sender().text()
                self.listWidget.setCurrentRow(index)

    def item_choice(self, item: QtWidgets.QListWidgetItem):
        """
        Triggered when an item in the category list is clicked.Will trigger radio button.

        Arguments:
            item: Item to be checked.
        """
        widget = self.listWidget.itemWidget(item)
        label_radio = widget.findChild(QtWidgets.QRadioButton, "label_radio")
        label_radio.setChecked(True)

    def update_current_group(self, text):
        """
        Update the current_group variable when the text in the QLineEdit changes.

        Arguments:
            text (str): The text of QLineEdit.
        """
        try:
            self.mainwindow.current_group = int(text)
        except ValueError:
            pass

    def increase_current_group(self):
        """Increase the current_group variable and update the QLineEdit text"""
        self.mainwindow.current_group += 1
        self.lineEdit_currentGroup.setText(str(self.mainwindow.current_group))

    def decrease_current_group(self):
        """Decrease the current_group variable and update the QLineEdit text"""
        if self.mainwindow.current_group > 1:
            self.mainwindow.current_group -= 1
            self.lineEdit_currentGroup.setText(str(self.mainwindow.current_group))

    def toggle_group_mode(self):
        """
        Toggle group mode.

        - auto: Group id auto add 1 when add a new polygon.
        - manual: Manual set group id.
        - track: Group id changed with the group of current polygons when use [TAB] or [`] to check.
        """
        _translate = QtCore.QCoreApplication.translate
        if self.mainwindow.group_select_mode == "auto":
            self.mainwindow.group_select_mode = "manual"
            self.pushButton_group_mode.setText("Manual")
            self.pushButton_group_mode.setStatusTip(
                _translate("MainWindow", "Manual set group id.")
            )
        elif self.mainwindow.group_select_mode == "manual":
            self.mainwindow.group_select_mode = "track"
            self.pushButton_group_mode.setText("Track")
            self.pushButton_group_mode.setStatusTip(
                _translate(
                    "MainWindow",
                    "Group id changed with the group of current polygons when use [TAB] or [`] to check.",
                )
            )
        elif self.mainwindow.group_select_mode == "track":
            self.mainwindow.group_select_mode = "auto"
            self.pushButton_group_mode.setText("Auto")
            self.pushButton_group_mode.setStatusTip(
                _translate("MainWindow", "Group id auto add 1 when add a new polygon.")
            )
