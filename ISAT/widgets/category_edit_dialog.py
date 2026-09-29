# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtCore, QtGui, QtWidgets

from ISAT.ui.category_edit import Ui_Dialog
from ISAT.widgets.undo_commands import (
    ShapeStateCommand,
    _snapshots_match,
    snapshot_shape,
)


class CategoryEditDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent, mainwindow, scene):
        super(CategoryEditDialog, self).__init__(parent)

        self.setupUi(self)
        self.mainwindow = mainwindow
        self.scene = scene
        self.polygons = []

        self.listWidget.itemClicked.connect(self.get_category)
        self.pushButton_apply.clicked.connect(self.apply)
        self.pushButton_cancel.clicked.connect(self.cancel)

        #
        self.checkBox_category_enabled.stateChanged.connect(self.check_category_enabled)
        self.checkBox_group_enabled.stateChanged.connect(self.check_group_enabled)
        self.checkBox_note_enabled.stateChanged.connect(self.check_note_enabled)
        self.checkBox_iscrowded_enabled.stateChanged.connect(self.check_crowded_enabled)

        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

    def check_category_enabled(self, checked):
        self.lineEdit_category.setEnabled(checked)

    def check_group_enabled(self, checked):
        self.spinBox_group.setEnabled(checked)

    def check_note_enabled(self, checked):
        self.lineEdit_note.setEnabled(checked)

    def check_crowded_enabled(self, checked):
        self.checkBox_iscrowded.setEnabled(checked)

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

            if len(self.polygons) == 1 and self.polygons[0].category == name:
                self.listWidget.setCurrentItem(item)

        if len(self.polygons) != 1:
            self.spinBox_group.clear()
            self.lineEdit_category.clear()
            self.checkBox_iscrowded.setCheckState(False)
            self.lineEdit_note.clear()
            self.label_layer.setText("{}".format(""))
            self.label_area.setText("{}".format(""))

            self.checkBox_category_enabled.setChecked(False)
            self.checkBox_group_enabled.setChecked(False)
            self.checkBox_note_enabled.setChecked(False)
            self.checkBox_iscrowded_enabled.setChecked(False)

        elif len(self.polygons) == 1:
            self.lineEdit_category.setText("{}".format(self.polygons[0].category))
            self.lineEdit_category.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.spinBox_group.setValue(self.polygons[0].group)
            iscrowd = (
                QtCore.Qt.CheckState.Checked
                if self.polygons[0].iscrowd
                else QtCore.Qt.CheckState.Unchecked
            )
            self.checkBox_iscrowded.setCheckState(iscrowd)
            self.lineEdit_note.setText("{}".format(self.polygons[0].note))
            self.label_layer.setText("{}".format(self.polygons[0].zValue()))
            self.label_area.setText(
                "{:.0f}{}".format(
                    self.polygons[0].area,
                    (
                        ""
                        if self.mainwindow.cfg["software"]["real_time_area"]
                        else "(no real time)"
                    ),
                )
            )

            self.checkBox_category_enabled.setChecked(True)
            self.checkBox_group_enabled.setChecked(True)
            self.checkBox_note_enabled.setChecked(True)
            self.checkBox_iscrowded_enabled.setChecked(True)

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
        """Set attributes of polygon (undoable).

        先整体解析目标属性并校验，再统一应用；应用时记录每个形状修改前/后的
        快照，作为一条（或多选时一个 macro 的）撤销步骤。
        """
        # 1) 解析目标属性（沿用原有“勾选才生效”的语义）
        targets = []
        for polygon in self.polygons:
            category = (
                self.lineEdit_category.text()
                if self.checkBox_category_enabled.isChecked()
                else polygon.category
            )
            if not category:
                QtWidgets.QMessageBox.warning(
                    self, "Warning", "Please select one category before submitting."
                )
                return
            group = (
                self.spinBox_group.value()
                if self.checkBox_group_enabled.isChecked()
                else polygon.group
            )
            is_crowd = (
                self.checkBox_iscrowded.isChecked()
                if self.checkBox_iscrowded_enabled.isChecked()
                else polygon.iscrowd
            )
            note = (
                self.lineEdit_note.text()
                if self.checkBox_note_enabled.isChecked()
                else polygon.note
            )
            targets.append((polygon, category, group, is_crowd, note))

        # 2) 应用，并记录 undo 快照（before 必须在 set_drawed 之前取）
        changed = []
        for polygon, category, group, is_crowd, note in targets:
            before = snapshot_shape(polygon)
            # 设置polygon 属性
            polygon.set_drawed(
                category,
                group,
                is_crowd,
                note,
                QtGui.QColor(self.mainwindow.category_color_dict.get(category, "#6F737A")),
            )
            after = snapshot_shape(polygon)
            if (
                before is not None
                and after is not None
                and not _snapshots_match(before, after)
            ):
                changed.append((polygon, before, after))

        # 3) 入撤销栈：单个直接 push，多选合成一步
        if changed:
            stack = self.mainwindow.undo_stack
            if len(changed) == 1:
                polygon, before, after = changed[0]
                stack.push(
                    ShapeStateCommand(
                        self.scene, "Edit attributes", polygon, before, after
                    )
                )
            else:
                stack.beginMacro("Edit attributes ({} shapes)".format(len(changed)))
                for polygon, before, after in changed:
                    stack.push(
                        ShapeStateCommand(
                            self.scene, "Edit attributes", polygon, before, after
                        )
                    )
                stack.endMacro()

        # 4) 刷新列表（原来在循环内刷新，多选时是 O(N²)，这里只刷一次）
        if targets:
            self.mainwindow.annos_dock_widget.update_listwidget()

        self.polygons = []
        self.scene.change_mode_to_view()
        self.close()

    def cancel(self):
        self.scene.cancel_draw()
        self.close()

    def closeEvent(self, a0: QtGui.QCloseEvent):
        self.cancel()

    def reject(self):
        self.cancel()
