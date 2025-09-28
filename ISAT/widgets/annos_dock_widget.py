# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtWidgets, QtCore
from ISAT.ui.anno_dock import Ui_Form
import functools
import re
from ISAT.widgets.polygon import Polygon


class AnnosDockWidget(QtWidgets.QWidget, Ui_Form):
    """
    Annotations dock widget.

    Attributes:
        mainwindow (ISAT.widgets.mainwindow.MainWindow): main window
        polygon_item_dict (dict): polygon and QListWidgetItem dict. {polygon: item}

    """

    def __init__(self, mainwindow):
        super(AnnosDockWidget, self).__init__()
        self.setupUi(self)
        self.mainwindow = mainwindow
        self.polygon_item_dict = {}

        self.listWidget.itemSelectionChanged.connect(self.set_polygon_selected)
        self.checkBox_visible.stateChanged.connect(self.set_all_polygon_visible)

        # addded group view
        self.comboBox_group_select.currentIndexChanged.connect(
            self.set_group_polygon_visible
        )
        self.button_next_group.clicked.connect(self.go_to_next_group)
        self.button_prev_group.clicked.connect(self.go_to_prev_group)

        self.listWidget.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.listWidget.customContextMenuRequested.connect(self.right_button_menu)

    def right_button_menu(self, point):
        self.mainwindow.right_button_menu.exec_(self.listWidget.mapToGlobal(point))

    def generate_item_and_itemwidget(
        self, polygon: Polygon
    ) -> (QtWidgets.QListWidgetItem, QtWidgets.QWidget):
        """
        The item of annotations list widget.

        Arguments:
            polygon (Polygon): polygon
        """
        color = self.mainwindow.category_color_dict.get(polygon.category, "#6F737A")
        item = QtWidgets.QListWidgetItem()
        item.setSizeHint(QtCore.QSize(200, 30))
        item_widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(9, 1, 9, 1)
        check_box = QtWidgets.QCheckBox()
        check_box.setFixedWidth(20)
        check_box.setChecked(polygon.isVisible())
        check_box.setObjectName("check_box")
        check_box.stateChanged.connect(
            functools.partial(self.set_polygon_show, polygon)
        )
        layout.addWidget(check_box)

        label_color = QtWidgets.QLabel()
        label_color.setFixedWidth(10)
        label_color.setStyleSheet("background-color: {};".format(color))
        layout.addWidget(label_color)

        category = QtWidgets.QLabel(polygon.category)

        group = QtWidgets.QLabel("{}".format(polygon.group))
        group.setFixedWidth(50)
        note = QtWidgets.QLabel("{}".format(polygon.note))
        note.setToolTip(polygon.note)
        note.setFixedWidth(46)

        label_iscrowd = QtWidgets.QLabel()
        label_iscrowd.setFixedWidth(3)
        if polygon.iscrowd == 1:
            label_iscrowd.setStyleSheet("background-color: {};".format("#000000"))

        layout.addWidget(category)
        layout.addWidget(group)
        layout.addWidget(note)
        layout.addWidget(label_iscrowd)

        item_widget.setLayout(layout)
        return item, item_widget

    def update_listwidget(self):
        """Update the annotations list widget."""
        current_group_id = self.comboBox_group_select.currentText()
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

        unique_groups = {polygon.group for polygon in self.mainwindow.polygons}
        self.comboBox_group_select.clear()
        self.comboBox_group_select.addItem("All")  # add an option to view all groups
        self.comboBox_group_select.addItems(
            sorted(
                [str(item) for item in unique_groups],
                key=lambda s: [
                    int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)
                ],
            )
        )
        if any(
            current_group_id == self.comboBox_group_select.itemText(i)
            for i in range(self.comboBox_group_select.count())
        ):
            self.comboBox_group_select.setCurrentText(current_group_id)
        else:
            self.comboBox_group_select.setCurrentIndex(0)

    def listwidget_add_polygon(self, polygon: Polygon):
        """
        Add a new item to the list widget.

        Arguments:
            polygon (Polygon): polygon
        """
        item, item_widget = self.generate_item_and_itemwidget(polygon)
        self.listWidget.addItem(item)
        self.listWidget.setItemWidget(item, item_widget)
        self.polygon_item_dict[polygon] = item
        self.mainwindow.set_saved_state(False)

    def listwidget_remove_polygon(self, polygon: Polygon):
        """
        Remove a item from the list widget.

        Arguments:
            polygon (Polygon): polygon
        """
        item = self.polygon_item_dict[polygon]
        self.listWidget.removeItemWidget(item)
        self.listWidget.takeItem(self.listWidget.row(item))
        del self.polygon_item_dict[polygon]
        self.mainwindow.set_saved_state(False)

    def set_selected(self, polygon: Polygon):
        """Set item selected."""
        item = self.polygon_item_dict[polygon]
        if polygon.isSelected():
            if not item.isSelected():
                item.setSelected(True)
                self.listWidget.setCurrentItem(item)
        if not polygon.isSelected():
            if item.isSelected():
                item.setSelected(False)

    def set_polygon_selected(self):
        """Set polygon selected on canvas."""
        items = self.listWidget.selectedItems()
        have_selected = True if items else False
        if have_selected:
            self.mainwindow.scene.change_mode_to_edit()

            if (
                len(self.mainwindow.scene.selected_polygons_list) != 2
                or len(self.listWidget.selectedItems()) != 2
            ):
                self.mainwindow.actionUnion.setEnabled(False)
                self.mainwindow.actionSubtract.setEnabled(False)
                self.mainwindow.actionIntersect.setEnabled(False)
                self.mainwindow.actionExclude.setEnabled(False)
            # 编辑，置顶等功能只针对单个多边形
            if len(items) > 1:
                self.mainwindow.actionTo_top.setEnabled(False)
                self.mainwindow.actionTo_bottom.setEnabled(False)
                self.mainwindow.actionEdit.setEnabled(False)
                self.mainwindow.actionCopy.setEnabled(False)
        else:
            self.mainwindow.scene.change_mode_to_view()

        for index, polygon in enumerate(self.mainwindow.polygons):
            if polygon not in self.polygon_item_dict:
                continue
            if self.polygon_item_dict[polygon] in items:
                if not polygon.isSelected():
                    polygon.setSelected(True)
            else:
                if polygon.isSelected():
                    polygon.setSelected(False)

    def set_polygon_show(self, polygon: Polygon):
        """
        Set polygon visible.

        Arguments:
            polygon (Polygon): polygon
        """
        for vertex in polygon.vertices:
            vertex.setVisible(self.sender().checkState())
        polygon.setVisible(self.sender().checkState())

    def set_all_polygon_visible(self, visible: bool = None):
        """
        Set all polygon visible.

        Arguments:
            visible (bool): visible flag
        """
        visible = self.checkBox_visible.isChecked() if visible is None else visible
        for index in range(self.listWidget.count()):
            item = self.listWidget.item(index)
            widget = self.listWidget.itemWidget(item)
            check_box = widget.findChild(QtWidgets.QCheckBox, "check_box")
            check_box.setChecked(visible)
        self.checkBox_visible.setChecked(visible)

    def set_group_polygon_visible(self):
        """
        Set group polygon visible.

        The group is self.comboBox_group_select.currentText()
        """
        selected_group = self.comboBox_group_select.currentText()

        for polygon, item in self.polygon_item_dict.items():
            widget = self.listWidget.itemWidget(item)
            check_box = widget.findChild(QtWidgets.QCheckBox, "check_box")
            if selected_group == "":
                return
            if selected_group == "All" or polygon.group == int(selected_group):
                check_box.setChecked(True)
            else:
                check_box.setChecked(False)

    def zoom_to_group(self):
        """Zoom to suitable size of group."""
        selected_group = self.comboBox_group_select.currentText()
        if selected_group == "":
            return
        if selected_group == "All":
            polygons_in_group = [
                polygon for polygon, item in self.polygon_item_dict.items()
            ]
        else:
            polygons_in_group = [
                polygon
                for polygon, item in self.polygon_item_dict.items()
                if polygon.group == int(selected_group)
            ]
        if not polygons_in_group:
            return
        min_x = min(
            min(vertex.x() for vertex in polygon.vertices)
            for polygon in polygons_in_group
        )
        min_y = min(
            min(vertex.y() for vertex in polygon.vertices)
            for polygon in polygons_in_group
        )
        max_x = max(
            max(vertex.x() for vertex in polygon.vertices)
            for polygon in polygons_in_group
        )
        max_y = max(
            max(vertex.y() for vertex in polygon.vertices)
            for polygon in polygons_in_group
        )
        margin = 20
        bounding_rect = QtCore.QRectF(
            min_x - margin,
            min_y - margin,
            max_x - min_x + 2 * margin,
            max_y - min_y + 2 * margin,
        )
        self.mainwindow.view.fitInView(
            bounding_rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio
        )

    def go_to_next_group(self):
        """Show next group."""
        current_index = self.comboBox_group_select.currentIndex()
        max_index = self.comboBox_group_select.count() - 1
        if current_index < max_index:
            self.comboBox_group_select.setCurrentIndex(current_index + 1)
            self.set_group_polygon_visible()
            self.zoom_to_group()

        if self.mainwindow.group_select_mode == "track":
            try:
                if self.comboBox_group_select.currentText() == "All":
                    max_group = self.comboBox_group_select.itemText(
                        len(self.comboBox_group_select) - 1
                    )
                    if max_group == "All":
                        max_group = 1
                    self.mainwindow.current_group = max_group
                    self.mainwindow.update_group_display()
                else:
                    group = int(self.comboBox_group_select.currentText())
                    self.mainwindow.current_group = group
                    self.mainwindow.update_group_display()
            except:
                pass

    def go_to_prev_group(self):
        """Show previous group."""
        current_index = self.comboBox_group_select.currentIndex()
        if current_index > 0:
            self.comboBox_group_select.setCurrentIndex(current_index - 1)
            self.set_group_polygon_visible()
            self.zoom_to_group()

        if self.mainwindow.group_select_mode == "track":
            try:
                if self.comboBox_group_select.currentText() == "All":
                    max_group = self.comboBox_group_select.itemText(
                        len(self.comboBox_group_select) - 1
                    )
                    if max_group == "All":
                        max_group = 1
                    self.mainwindow.current_group = max_group
                    self.mainwindow.update_group_display()
                else:
                    group = int(self.comboBox_group_select.currentText())
                    self.mainwindow.current_group = group
                    self.mainwindow.update_group_display()
            except:
                pass
