# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtCore, QtGui, QtWidgets

from ISAT.ui.category_edit import Ui_Dialog


class CategoryEditDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent, mainwindow, scene):
        super(CategoryEditDialog, self).__init__(parent)

        self.setupUi(self)
        self.mainwindow = mainwindow
        self.scene = scene
        self.polygon = None

        self.listWidget.itemClicked.connect(self.get_category)
        self.pushButton_apply.clicked.connect(self.apply)
        self.pushButton_cancel.clicked.connect(self.cancel)

        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

    def load_cfg(self):
        """Load the cfg and update the interface."""
        self.listWidget.clear()

        labels = self.mainwindow.cfg.get("label", [])

        for label in labels:
            name = label.get("name", "UNKNOW")
            color = label.get("color", "#000000")
            # item = QtWidgets.QListWidgetItem()
            # item.setText(name)
            # item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            # self.listWidget.addItem(item)

            item = QtWidgets.QListWidgetItem()
            item.setSizeHint(QtCore.QSize(200, 30))
            widget = QtWidgets.QWidget()

            layout = QtWidgets.QHBoxLayout()
            layout.setContentsMargins(9, 1, 9, 1)
            label_category = QtWidgets.QLabel()
            label_category.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            label_category.setText(name)
            label_category.setObjectName("label_category")

            label_color = QtWidgets.QLabel()
            label_color.setFixedWidth(10)
            label_color.setStyleSheet("background-color: {};".format(color))
            label_color.setObjectName("label_color")

            layout.addWidget(label_color)
            layout.addWidget(label_category)
            widget.setLayout(layout)

            self.listWidget.addItem(item)
            self.listWidget.setItemWidget(item, widget)

            if self.polygon is not None and self.polygon.category == name:
                self.listWidget.setCurrentItem(item)

        if self.polygon is None:
            self.spinBox_group.clear()
            self.lineEdit_category.clear()
            self.checkBox_iscrowded.setCheckState(False)
            self.lineEdit_note.clear()
            self.label_layer.setText("{}".format(""))
            self.label_area.setText("{}".format(""))
        else:
            self.lineEdit_category.setText("{}".format(self.polygon.category))
            self.lineEdit_category.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.spinBox_group.setValue(self.polygon.group)
            iscrowd = (
                QtCore.Qt.CheckState.Checked
                if self.polygon.iscrowd
                else QtCore.Qt.CheckState.Unchecked
            )
            self.checkBox_iscrowded.setCheckState(iscrowd)
            self.lineEdit_note.setText("{}".format(self.polygon.note))
            self.label_layer.setText("{}".format(self.polygon.zValue()))
            self.label_area.setText(
                "{:.0f}{}".format(
                    self.polygon.area,
                    (
                        ""
                        if self.mainwindow.cfg["software"]["real_time_area"]
                        else "(no real time)"
                    ),
                )
            )
        if self.listWidget.count() == 0:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "Please set categorys before tagging."
            )

    def get_category(self, item: QtWidgets.QListWidgetItem):
        """
        Triggered when category item selected.

        Arguments:
            item: category item.
        """
        widget = self.listWidget.itemWidget(item)
        label_category = widget.findChild(QtWidgets.QLabel, "label_category")
        self.lineEdit_category.setText(label_category.text())
        self.lineEdit_category.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

    def apply(self):
        """Set attributes of polygon."""
        category = self.lineEdit_category.text()
        group = self.spinBox_group.value()

        is_crowd = self.checkBox_iscrowded.isChecked()
        note = self.lineEdit_note.text()
        if not category:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "Please select one category before submitting."
            )
            return

        # 设置polygon 属性
        self.polygon.set_drawed(
            category,
            group,
            is_crowd,
            note,
            QtGui.QColor(self.mainwindow.category_color_dict.get(category, "#6F737A")),
        )
        self.mainwindow.annos_dock_widget.update_listwidget()

        self.polygon = None
        self.scene.change_mode_to_view()
        self.close()

    def cancel(self):
        self.scene.cancel_draw()
        self.close()

    def closeEvent(self, a0: QtGui.QCloseEvent):
        self.cancel()

    def reject(self):
        self.cancel()
