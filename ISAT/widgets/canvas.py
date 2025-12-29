# -*- coding: utf-8 -*-
# @Author  : LG
import math  # 用于角度约束计算
import time  # 拖动鼠标描点

import cv2
import numpy as np
import shapely
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets

from ISAT.configs import CONTOURMode, CONTOURMethod, DRAWMode, STATUSMode
from ISAT.utils.dicom import load_dcm_as_image
from ISAT.widgets.polygon import Line, Polygon, PromptPoint, Rect, Vertex


class AnnotationScene(QtWidgets.QGraphicsScene):
    """
    Annotation Scene.

    Arguments:
        mainwindow (ISAT.widgets.mainwindow.MainWindow): ISAT main window

    Attributes:
        image_item (QtWidgets.QGraphicsPixmapItem): Image pixmap item.
        mask_item (QtWidgets.QGraphicsPixmapItem): SAM mask pixmap item.
        image_data (np.ndarray): Image data.
        current_graph (Polygon): The polygon being annotated.
        prompt_box_item (Rect): The box for SAM box prompt.
        repaint_line_item (Line): The line for repaint mode.
        mode (STATUSMode): STATUSMode. eg: CREATE, VIEW, EDIT, REPAINT.
        draw_mode (ISAT.configs.DRAWMode): draw mode.eg:POLYGON, SEGMENTANYTHING_POINT, SEGMENTANYTHING_BOX
        contour_mode (CONTOURMode): The mode for convert sam mask to polygon.
        prompt_point_positions (list): The click points for sam point prompt.
        prompt_point_labels (list): The tag for sam point prompt.
        prompt_point_items (list[PromptPoint, ...]): The prompt points for sam point prompt.
        mask (np.ndarray): The mask output by sam model.
        mask_alpha (float): The alpha value for the mask.
        guide_line_x (QtWidgets.QGraphicsLineItem): The guideline.
        guide_line_y (QtWidgets.QGraphicsLineItem):The guideline.

        last_draw_time: Counter for the dragging annotation method, used by draw polygon and repaint.
        draw_interval (float): Interval for the dragging annotation method.
        pressed (bool): The tag for mouse pressed.

        selected_polygons_list (list): The list of polygons selected.

        repaint_start_vertex (Vertex): The start vertex for repaint.
        repaint_end_vertex (Vertex): The end vertex for repaint.
        hovered_vertex (Vertex): The hovered vertex for repaint.
    """

    def __init__(self, mainwindow):
        super(AnnotationScene, self).__init__()
        self.mainwindow = mainwindow
        self.image_item: QtWidgets.QGraphicsPixmapItem = None
        self.mask_item: QtWidgets.QGraphicsPixmapItem = None
        self.image_data = None
        self.current_graph: Polygon = None

        self.mode = STATUSMode.VIEW
        self.draw_mode = DRAWMode.SEGMENTANYTHING_POINT   # 默认使用segment anything进行快速标注
        self.contour_mode = CONTOURMode.SAVE_EXTERNAL   # 默认SAM只保留外轮廓
        self.contour_method = CONTOURMethod.SIMPLE  # 默认使用Simple

        # for point prompt
        self.prompt_point_positions = []
        self.prompt_point_labels = []
        self.prompt_point_items = []

        # for box prompt
        self.prompt_box_item: Rect = None

        # for visual prompt
        self.prompt_visual_current_item: Rect = None  # 当前正在绘制的矩形
        self.prompt_visual_current_label: bool = True # 当前正在绘制的矩形类型
        self.prompt_visual_items = []   # 存储已添加的矩形item
        self.prompt_visual_labels = []  # 存储已添加的矩形类型

        # repaint line item
        self.repaint_line_item: Line = None

        self.mask: np.ndarray = None
        self.mask_alpha = 0.5

        self.guide_line_x: QtWidgets.QGraphicsLineItem = None
        self.guide_line_y: QtWidgets.QGraphicsLineItem = None

        # 拖动鼠标描点
        self.last_draw_time = time.time()
        self.draw_interval = 0.15
        self.pressed = False

        # 按键状态
        self.shift_pressed = False
        self.ctrl_pressed = False

        #
        self.selected_polygons_list = list()

        self.repaint_start_vertex = None
        self.repaint_end_vertex = None
        self.hovered_vertex: Vertex = None

    def load_image(self, image_path: str):
        """
        Load image.

        :param image_path: The image path.
        :return:
        """
        self.clear()
        if self.mainwindow.use_segment_anything:
            self.mainwindow.segany.reset_image()

        if image_path.lower().endswith(".dcm"):
            image = load_dcm_as_image(image_path)
        else:
            image = Image.open(image_path)
        if self.mainwindow.can_be_annotated:
            image = image.convert("RGB")
        self.image_data = np.array(image)
        self.image_item = QtWidgets.QGraphicsPixmapItem()
        self.image_item.setZValue(0)
        self.addItem(self.image_item)
        self.mask_item = QtWidgets.QGraphicsPixmapItem()
        self.mask_item.setZValue(1)
        self.addItem(self.mask_item)

        height, width, channel = self.image_data.shape
        bytes_per_line = channel * width
        q_image = QtGui.QImage(
            self.image_data.tobytes(),
            width,
            height,
            bytes_per_line,
            QtGui.QImage.Format_RGB888,
        )
        self.image_item.setPixmap(QtGui.QPixmap.fromImage(q_image))
        # self.image_item.setPixmap(QtGui.QPixmap(image_path))
        self.setSceneRect(self.image_item.boundingRect())
        self.change_mode_to_view()

    def unload_image(self):
        """Unload image and clear scene."""
        self.clear()
        self.setSceneRect(QtCore.QRectF())
        self.mainwindow.polygons.clear()
        self.image_item = None
        self.mask_item = None
        self.current_graph = None

    def change_mode_to_create(self):
        """Change to create mode."""
        if self.image_item is None:
            return
        self.mode = STATUSMode.CREATE
        if self.image_item is not None:
            self.image_item.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
        self.mainwindow.actionPrev_image.setEnabled(False)
        self.mainwindow.actionNext_image.setEnabled(False)

        self.mainwindow.actionSegment_anything_point.setEnabled(False)
        self.mainwindow.actionSegment_anything_box.setEnabled(False)
        self.mainwindow.actionPolygon.setEnabled(False)
        self.mainwindow.actionBackspace.setEnabled(True)
        self.mainwindow.actionFinish.setEnabled(True)
        self.mainwindow.actionCancel.setEnabled(True)

        self.mainwindow.actionTo_top.setEnabled(False)
        self.mainwindow.actionTo_bottom.setEnabled(False)
        self.mainwindow.actionEdit.setEnabled(False)
        self.mainwindow.actionCopy.setEnabled(False)
        self.mainwindow.actionUnion.setEnabled(False)
        self.mainwindow.actionSubtract.setEnabled(False)
        self.mainwindow.actionIntersect.setEnabled(False)
        self.mainwindow.actionExclude.setEnabled(False)
        self.mainwindow.actionDelete.setEnabled(False)
        self.mainwindow.actionSave.setEnabled(False)
        self.mainwindow.actionVisible.setEnabled(True)

        self.mainwindow.annos_dock_widget.setEnabled(False)
        self.mainwindow.actionRepaint.setEnabled(False)

        self.mainwindow.modeState.setText("C")
        self.mainwindow.modeState.setStatusTip(
            QtCore.QCoreApplication.translate("MainWindow", "Create mode.")
        )
        self.mainwindow.modeState.setStyleSheet(
            """
            background-color: #6CAB74;
            border-radius : 5px; 
            color: white;
        """
        )

    def change_mode_to_view(self):
        """Change to view mode."""
        self.mode = STATUSMode.VIEW
        if self.image_item is not None:
            self.image_item.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.ArrowCursor))

        self.mainwindow.actionPrev_image.setEnabled(True)
        self.mainwindow.actionNext_image.setEnabled(True)
        self.mainwindow.SeganyEnabled()
        self.mainwindow.actionPolygon.setEnabled(self.mainwindow.can_be_annotated)
        self.mainwindow.actionBackspace.setEnabled(False)
        self.mainwindow.actionFinish.setEnabled(False)
        self.mainwindow.actionCancel.setEnabled(True)

        self.mainwindow.actionTo_top.setEnabled(False)
        self.mainwindow.actionTo_bottom.setEnabled(False)
        self.mainwindow.actionEdit.setEnabled(False)
        self.mainwindow.actionCopy.setEnabled(False)
        self.mainwindow.actionUnion.setEnabled(False)
        self.mainwindow.actionSubtract.setEnabled(False)
        self.mainwindow.actionIntersect.setEnabled(False)
        self.mainwindow.actionExclude.setEnabled(False)
        self.mainwindow.actionDelete.setEnabled(False)
        self.mainwindow.actionSave.setEnabled(self.mainwindow.can_be_annotated)
        self.mainwindow.actionVisible.setEnabled(True)
        self.mainwindow.actionRepaint.setEnabled(True)

        self.mainwindow.annos_dock_widget.setEnabled(True)

        self.mainwindow.modeState.setText("V")
        self.mainwindow.modeState.setStatusTip(
            QtCore.QCoreApplication.translate("MainWindow", "View mode.")
        )
        self.mainwindow.modeState.setStyleSheet(
            """
            background-color: #70AEFF;
            border-radius : 5px; 
            color: white;
        """
        )

    def change_mode_to_edit(self):
        """Change to edit mode."""
        self.mode = STATUSMode.EDIT
        if self.image_item is not None:
            self.image_item.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))

        self.mainwindow.actionPrev_image.setEnabled(False)
        self.mainwindow.actionNext_image.setEnabled(False)

        self.mainwindow.actionSegment_anything_point.setEnabled(False)
        self.mainwindow.actionSegment_anything_box.setEnabled(False)
        self.mainwindow.actionPolygon.setEnabled(False)
        self.mainwindow.actionBackspace.setEnabled(False)
        self.mainwindow.actionFinish.setEnabled(False)
        self.mainwindow.actionCancel.setEnabled(True)

        self.mainwindow.actionTo_top.setEnabled(True)
        self.mainwindow.actionTo_bottom.setEnabled(True)
        self.mainwindow.actionEdit.setEnabled(True)
        self.mainwindow.actionCopy.setEnabled(True)
        self.mainwindow.actionUnion.setEnabled(True)
        self.mainwindow.actionSubtract.setEnabled(True)
        self.mainwindow.actionIntersect.setEnabled(True)
        self.mainwindow.actionExclude.setEnabled(True)
        self.mainwindow.actionDelete.setEnabled(True)
        self.mainwindow.actionSave.setEnabled(True)
        self.mainwindow.actionVisible.setEnabled(True)
        self.mainwindow.actionRepaint.setEnabled(False)

        self.mainwindow.modeState.setText("E")
        self.mainwindow.modeState.setStatusTip(
            QtCore.QCoreApplication.translate("MainWindow", "Edit mode.")
        )
        self.mainwindow.modeState.setStyleSheet(
            """
            background-color: #51C0CF;
            border-radius : 5px; 
            color: white;
        """
        )

    def change_mode_to_repaint(self):
        """Change to repaint mode."""
        self.mode = STATUSMode.REPAINT
        self.repaint_start_vertex = None
        self.repaint_end_vertex = None
        if self.repaint_line_item is None:
            self.repaint_line_item = Line()  # 重绘部分，由起始点开始的线段显示
            self.addItem(self.repaint_line_item)

        if self.image_item is not None:
            self.image_item.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))

        self.mainwindow.actionPrev_image.setEnabled(False)
        self.mainwindow.actionNext_image.setEnabled(False)

        self.mainwindow.actionSegment_anything_point.setEnabled(False)
        self.mainwindow.actionSegment_anything_box.setEnabled(False)
        self.mainwindow.actionPolygon.setEnabled(False)
        self.mainwindow.actionBackspace.setEnabled(True)
        self.mainwindow.actionFinish.setEnabled(False)
        self.mainwindow.actionCancel.setEnabled(True)

        self.mainwindow.actionTo_top.setEnabled(False)
        self.mainwindow.actionTo_bottom.setEnabled(False)
        self.mainwindow.actionEdit.setEnabled(False)
        self.mainwindow.actionCopy.setEnabled(False)
        self.mainwindow.actionUnion.setEnabled(False)
        self.mainwindow.actionSubtract.setEnabled(False)
        self.mainwindow.actionIntersect.setEnabled(False)
        self.mainwindow.actionExclude.setEnabled(False)
        self.mainwindow.actionDelete.setEnabled(False)
        self.mainwindow.actionSave.setEnabled(True)
        self.mainwindow.actionVisible.setEnabled(False)
        self.mainwindow.actionRepaint.setEnabled(False)

        self.mainwindow.modeState.setText("R")
        self.mainwindow.modeState.setStatusTip(
            QtCore.QCoreApplication.translate("MainWindow", "Repaint mode.")
        )
        self.mainwindow.modeState.setStyleSheet(
            """
            background-color: #CF84CF;
            border-radius : 5px; 
            color: white;
        """
        )

    def change_contour_mode_to_save_all(self):
        """Change to save all contour mode for convert mask to polygons."""
        self.contour_mode = CONTOURMode.SAVE_ALL

    def change_contour_mode_to_save_max_only(self):
        """Change to save max contour mode for convert mask to polygons."""
        self.contour_mode = CONTOURMode.SAVE_MAX_ONLY

    def change_contour_mode_to_save_external(self):
        """Change to save external contour mode for convert mask to polygons."""
        self.contour_mode = CONTOURMode.SAVE_EXTERNAL

    def change_contour_method_to_simple(self):
        self.contour_method = CONTOURMethod.SIMPLE

    def change_contour_method_to_tc89_kcos(self):
        self.contour_method = CONTOURMethod.TC89_KCOS

    def change_contour_method_to_none(self):
        self.contour_method = CONTOURMethod.NONE

    def start_segment_anything(self):
        """Start segmenting anything with point prompt."""
        self.draw_mode = DRAWMode.SEGMENTANYTHING_POINT
        self.start_draw()

    def start_segment_anything_box(self):
        """Start segmenting anything with box prompt."""
        if self.prompt_box_item is not None:
            try:
                self.prompt_box_item.delete()
                self.removeItem(self.prompt_box_item)
            finally:
                self.prompt_box_item = None

        self.prompt_box_item = Rect()
        self.prompt_box_item.setZValue(2)
        self.addItem(self.prompt_box_item)

        self.draw_mode = DRAWMode.SEGMENTANYTHING_BOX
        self.start_draw()

    def start_segment_anything_visual(self, positive: bool=True):
        """Start segmenting anything with visual prompt."""
        if self.mainwindow.current_index is None:
            return

        if self.prompt_visual_current_item is not None:
            try:
                self.prompt_visual_current_item.delete()
                self.removeItem(self.prompt_visual_current_item)
            finally:
                self.prompt_visual_current_item = None

        self.prompt_visual_current_item = Rect()
        self.prompt_visual_current_item.setZValue(2)
        pen = QtGui.QPen(QtGui.QColor("#00ff00" if positive else "#ff0000"))
        pen.setStyle(QtCore.Qt.PenStyle.DotLine)
        self.prompt_visual_current_item.setPen(pen)
        self.prompt_visual_current_label = positive
        self.addItem(self.prompt_visual_current_item)

        self.draw_mode = DRAWMode.SEGMENTANYTHING_VISUAL
        self.start_draw()

    def start_draw_polygon(self):
        """Start drawing polygon."""
        self.draw_mode = DRAWMode.POLYGON
        self.start_draw()

    def start_draw(self):
        """Try change to create mode and add a empty polygon for annotation ops."""
        # 只有view模式时，才能切换create模式
        if self.mode != STATUSMode.VIEW:
            return

        self.mainwindow.plugin_manager_dialog.trigger_before_annotation_start()

        # 否则，切换到绘图模式
        self.change_mode_to_create()
        if self.mainwindow.cfg["software"]["create_mode_invisible_polygon"]:
            self.mainwindow.set_labels_visible(False)

        # 绘图模式
        if self.mode == STATUSMode.CREATE:
            self.current_graph = Polygon()
            self.current_graph.hover_alpha = int(
                self.mainwindow.cfg["software"]["polygon_alpha_hover"] * 255
            )
            self.current_graph.nohover_alpha = int(
                self.mainwindow.cfg["software"]["polygon_alpha_no_hover"] * 255
            )
            color = self.current_graph.color
            color.setAlpha(self.current_graph.hover_alpha)
            self.current_graph.setBrush(color)
            self.addItem(self.current_graph)

    def finish_draw(self):
        """Finish annotation. Convert mask to polygons when using sam, if the number of vertices less than 3, delete polygon."""

        category = self.mainwindow.current_category
        group = self.mainwindow.current_group
        is_crowd = False
        note = ""

        if (
            self.draw_mode == DRAWMode.SEGMENTANYTHING_POINT
            or self.draw_mode == DRAWMode.SEGMENTANYTHING_BOX
        ):
            # mask to polygon
            # --------------
            if self.mask is not None:
                mask = self.mask

                contours, hierarchy = self.mainwindow.mask_to_polygon(mask)

                for index, contour in enumerate(contours):

                    if self.current_graph is None:
                        self.current_graph = Polygon()
                        self.addItem(self.current_graph)

                    self.current_graph.hover_alpha = int(
                        self.mainwindow.cfg["software"]["polygon_alpha_hover"] * 255
                    )
                    self.current_graph.nohover_alpha = int(
                        self.mainwindow.cfg["software"]["polygon_alpha_no_hover"] * 255
                    )

                    if len(contour) < 3:
                        continue
                    for point in contour:
                        x, y = point[0]
                        x = max(0.1, x)
                        y = max(0.1, y)
                        self.current_graph.addPoint(QtCore.QPointF(x, y))

                    if (
                        self.contour_mode == CONTOURMode.SAVE_ALL
                        and hierarchy[0][index][3] != -1
                    ):
                        # 保存所有轮廓，且当前轮廓为子轮廓，则自轮廓类别设置为背景
                        category = "__background__"
                        group = 0
                    else:
                        category = self.mainwindow.current_category
                        group = self.mainwindow.current_group

                    self.current_graph.set_drawed(
                        category,
                        group,
                        is_crowd,
                        note,
                        QtGui.QColor(
                            self.mainwindow.category_color_dict.get(category, "#6F737A")
                        ),
                        len(self.mainwindow.polygons) + 1,
                    )

                    # 添加新polygon
                    self.mainwindow.polygons.append(self.current_graph)
                    self.mainwindow.annos_dock_widget.listwidget_add_polygon(
                        self.current_graph
                    )

                    self.current_graph = None
                if self.mainwindow.group_select_mode == "auto":
                    self.mainwindow.current_group += 1
                    self.mainwindow.categories_dock_widget.lineEdit_currentGroup.setText(
                        str(self.mainwindow.current_group)
                    )
                self.mask = None
        elif self.draw_mode == DRAWMode.POLYGON:
            if self.current_graph is None or len(self.current_graph.points) < 1:
                return

            # 移除鼠标移动点
            self.current_graph.removePoint(len(self.current_graph.points) - 1)

            # 单点，删除
            if len(self.current_graph.points) < 2:
                self.current_graph.delete()
                self.removeItem(self.current_graph)

                self.change_mode_to_view()
                if self.mainwindow.cfg["software"]["create_mode_invisible_polygon"]:
                    self.mainwindow.set_labels_visible(True)

                return

            # 两点，默认矩形
            if len(self.current_graph.points) == 2:
                first_point = self.current_graph.points[0]
                last_point = self.current_graph.points[-1]
                self.current_graph.removePoint(len(self.current_graph.points) - 1)
                self.current_graph.addPoint(
                    QtCore.QPointF(first_point.x(), last_point.y())
                )
                self.current_graph.addPoint(last_point)
                self.current_graph.addPoint(
                    QtCore.QPointF(last_point.x(), first_point.y())
                )

            # 设置polygon 属性
            self.current_graph.set_drawed(
                category,
                group,
                is_crowd,
                note,
                QtGui.QColor(
                    self.mainwindow.category_color_dict.get(category, "#6F737A")
                ),
                len(self.mainwindow.polygons) + 1,
            )
            if self.mainwindow.group_select_mode == "auto":
                self.mainwindow.current_group += 1
                self.mainwindow.categories_dock_widget.lineEdit_currentGroup.setText(
                    str(self.mainwindow.current_group)
                )

            # 添加新polygon
            self.mainwindow.polygons.append(self.current_graph)
            self.mainwindow.annos_dock_widget.listwidget_add_polygon(self.current_graph)

        # 选择类别
        # self.mainwindow.category_choice_widget.load_cfg()
        # self.mainwindow.category_choice_widget.show()

        self.current_graph = None

        # prompt box clear
        if self.prompt_box_item is not None:
            self.prompt_box_item.delete()
            self.removeItem(self.prompt_box_item)
            self.prompt_box_item = None

        # prompt point clear
        self.prompt_point_positions.clear()
        self.prompt_point_labels.clear()
        for prompt_point_item in self.prompt_point_items:
            try:
                self.removeItem(prompt_point_item)
            finally:
                del prompt_point_item
        self.prompt_point_items.clear()

        # visual prompt clear
        if self.prompt_visual_current_item is not None:
            self.prompt_visual_current_item.delete()
            self.removeItem(self.prompt_visual_current_item)
            self.prompt_visual_current_item = None
        for prompt_visual_item in self.prompt_visual_items:
            try:
                prompt_visual_item.delete()
                self.removeItem(prompt_visual_item)
            finally:
                del prompt_visual_item
        self.prompt_visual_items.clear()
        self.prompt_visual_labels.clear()

        self.change_mode_to_view()
        if self.mainwindow.cfg["software"]["create_mode_invisible_polygon"]:
            self.mainwindow.set_labels_visible(True)

        self.update_mask()

        self.mainwindow.plugin_manager_dialog.trigger_after_annotation_created()

    def cancel_draw(self):
        """Cancel draw. Remove the drawing polygons and masks, prompt points, prompt box eg."""
        if self.mode == STATUSMode.CREATE:
            if self.current_graph is not None:
                self.current_graph.delete()  # 清除所有路径
                self.removeItem(self.current_graph)
                self.current_graph = None
        if self.mode == STATUSMode.REPAINT:
            if self.repaint_line_item is not None:
                self.repaint_line_item.delete()
                self.removeItem(self.repaint_line_item)
                self.repaint_line_item = None
        if self.mode == STATUSMode.EDIT:
            for item in self.selectedItems():
                item.setSelected(False)

        # prompt box clear
        if self.prompt_box_item is not None:
            self.prompt_box_item.delete()
            self.removeItem(self.prompt_box_item)
            self.prompt_box_item = None

        # prompt point clear
        self.prompt_point_positions.clear()
        self.prompt_point_labels.clear()
        for prompt_point_item in self.prompt_point_items:
            try:
                self.removeItem(prompt_point_item)
            finally:
                del prompt_point_item
        self.prompt_point_items.clear()

        # visual prompt clear
        if self.prompt_visual_current_item is not None:
            self.prompt_visual_current_item.delete()
            self.removeItem(self.prompt_visual_current_item)
            self.prompt_visual_current_item = None
        for prompt_visual_item in self.prompt_visual_items:
            try:
                prompt_visual_item.delete()
                self.removeItem(prompt_visual_item)
            finally:
                del prompt_visual_item
        self.prompt_visual_items.clear()
        self.prompt_visual_labels.clear()

        self.change_mode_to_view()
        if self.mainwindow.cfg["software"]["create_mode_invisible_polygon"]:
            self.mainwindow.set_labels_visible(True)

        self.update_mask()

    def delete_selected_graph(self):
        """Delete selected graph. Graph can be polygons or vertices, support multiple selection modes by pressing the CTRL key."""
        deleted_layer = None
        for item in self.selectedItems():
            if isinstance(item, Polygon) and (item in self.mainwindow.polygons):
                if item in self.selected_polygons_list:
                    self.selected_polygons_list.remove(item)
                self.mainwindow.polygons.remove(item)
                self.mainwindow.annos_dock_widget.listwidget_remove_polygon(item)
                item.delete()
                self.removeItem(item)
                deleted_layer = item.zValue()
                del item
            elif isinstance(item, Vertex):
                polygon = item.polygon
                if polygon.vertices:
                    index = polygon.vertices.index(item)
                    item.polygon.removePoint(index)
                else:
                    self.removeItem(item)
                    del item
                # 如果剩余顶点少于三个，删除多边形
                if len(polygon.vertices) < 3:
                    if polygon in self.mainwindow.polygons:
                        self.mainwindow.polygons.remove(polygon)
                        self.mainwindow.annos_dock_widget.listwidget_remove_polygon(
                            polygon
                        )
                        polygon.delete()
                    if polygon in self.items():
                        self.removeItem(polygon)
                    deleted_layer = polygon.zValue()
                    del polygon

        if deleted_layer is not None:
            for p in self.mainwindow.polygons:
                if p.zValue() > deleted_layer:
                    p.setZValue(p.zValue() - 1)

    def edit_polygon(self):
        """Edit the selected polygon. Open edit window then edit the attributes of the polygon."""
        selectd_items = self.selectedItems()
        selectd_items = [item for item in selectd_items if isinstance(item, Polygon)]
        if len(selectd_items) < 1:
            return
        item = selectd_items[0]
        if not item:
            return
        self.mainwindow.category_edit_widget.polygon = item
        self.mainwindow.category_edit_widget.load_cfg()
        self.mainwindow.category_edit_widget.show()

    def move_polygon_to_top(self):
        """Move the selected polygon to top layer."""
        selectd_items = self.selectedItems()
        selectd_items = [item for item in selectd_items if isinstance(item, Polygon)]
        if len(selectd_items) < 1:
            return
        current_polygon = selectd_items[0]
        max_layer = len(self.mainwindow.polygons)

        current_layer = current_polygon.zValue()
        for p in self.mainwindow.polygons:
            if p.zValue() > current_layer:
                p.setZValue(p.zValue() - 1)

        current_polygon.setZValue(max_layer)
        for vertex in current_polygon.vertices:
            vertex.setZValue(max_layer)
        self.mainwindow.set_saved_state(False)

    def move_polygon_to_bottom(self):
        """Move the selected polygon to bottom layer."""
        selectd_items = self.selectedItems()
        selectd_items = [item for item in selectd_items if isinstance(item, Polygon)]

        if len(selectd_items) < 1:
            return
        current_polygon = selectd_items[0]

        if current_polygon is not None:
            current_layer = current_polygon.zValue()

            for p in self.mainwindow.polygons:
                if p.zValue() < current_layer:
                    p.setZValue(p.zValue() + 1)

            current_polygon.setZValue(1)
            for vertex in current_polygon.vertices:
                vertex.setZValue(1)
        self.mainwindow.set_saved_state(False)

    def copy_item(self):
        """Copy selected polygon. The copied polygon has the sam attributes with ori polygon."""
        for item in self.selectedItems():
            if isinstance(item, Polygon):
                index = self.mainwindow.polygons.index(item)
                if self.current_graph is None:
                    self.current_graph = Polygon()
                    self.addItem(self.current_graph)

                self.current_graph.hover_alpha = int(
                    self.mainwindow.cfg["software"]["polygon_alpha_hover"] * 255
                )
                self.current_graph.nohover_alpha = int(
                    self.mainwindow.cfg["software"]["polygon_alpha_no_hover"] * 255
                )

                for point in item.vertices:
                    x, y = point.x(), point.y()
                    self.current_graph.addPoint(QtCore.QPointF(x, y))

                self.current_graph.set_drawed(
                    item.category,
                    item.group,
                    item.iscrowd,
                    item.note,
                    item.color,
                    int(item.zValue()),
                )
                self.mainwindow.polygons.insert(index, self.current_graph)
                self.mainwindow.annos_dock_widget.listwidget_add_polygon(
                    self.current_graph
                )
                item.setSelected(False)
                self.current_graph.setSelected(True)
                self.current_graph = None

    # 感谢[XieDeWu](https://github.com/XieDeWu)提的有关交、并、差、异或的[建议](https://github.com/yatengLG/ISAT_with_segment_anything/issues/167)。
    def polygons_union(self):
        """Union. Only support two polygons. Always use the attributes of the first polygon."""
        if len(self.selected_polygons_list) == 2:
            index = self.mainwindow.polygons.index(self.selected_polygons_list[0])

            category = self.selected_polygons_list[0].category
            group = self.selected_polygons_list[0].group
            iscrowd = self.selected_polygons_list[0].iscrowd
            note = self.selected_polygons_list[0].note
            layer = self.selected_polygons_list[0].zValue()
            color = self.selected_polygons_list[0].color

            try:
                polygon1_shapely = shapely.Polygon(
                    [
                        (point.x(), point.y())
                        for point in self.selected_polygons_list[0].vertices
                    ]
                )
                polygon2_shapely = shapely.Polygon(
                    [
                        (point.x(), point.y())
                        for point in self.selected_polygons_list[1].vertices
                    ]
                )
                return_shapely = polygon1_shapely.union(polygon2_shapely)
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self.mainwindow, "Warning", "Polygon warning: {}".format(e)
                )
                return

            if isinstance(return_shapely, shapely.Polygon):

                # 创建新多边形
                if self.current_graph is None:
                    self.current_graph = Polygon()
                    self.addItem(self.current_graph)

                self.current_graph.hover_alpha = int(
                    self.mainwindow.cfg["software"]["polygon_alpha_hover"] * 255
                )
                self.current_graph.nohover_alpha = int(
                    self.mainwindow.cfg["software"]["polygon_alpha_no_hover"] * 255
                )

                for point in return_shapely.exterior.coords:
                    x, y = point[0], point[1]
                    self.current_graph.addPoint(QtCore.QPointF(x, y))

                self.current_graph.set_drawed(
                    category, group, iscrowd, note, color, layer
                )
                self.mainwindow.polygons.insert(index, self.current_graph)
                self.current_graph = None

                # 删除旧的多边形
                for polygon_item in self.selected_polygons_list:
                    self.mainwindow.polygons.remove(polygon_item)
                    polygon_item.delete()
                    self.removeItem(polygon_item)
                    del polygon_item
                self.selected_polygons_list.clear()

                self.mainwindow.annos_dock_widget.update_listwidget()

    def polygons_difference(self):
        """Subtract. Only support two polygons. Always use the attributes of the first polygon."""
        if len(self.selected_polygons_list) == 2:
            index = self.mainwindow.polygons.index(self.selected_polygons_list[0])

            category = self.selected_polygons_list[0].category
            group = self.selected_polygons_list[0].group
            iscrowd = self.selected_polygons_list[0].iscrowd
            note = self.selected_polygons_list[0].note
            layer = self.selected_polygons_list[0].zValue()
            color = self.selected_polygons_list[0].color
            try:
                polygon1_shapely = shapely.Polygon(
                    [
                        (point.x(), point.y())
                        for point in self.selected_polygons_list[0].vertices
                    ]
                )
                polygon2_shapely = shapely.Polygon(
                    [
                        (point.x(), point.y())
                        for point in self.selected_polygons_list[1].vertices
                    ]
                )
                return_shapely = polygon1_shapely.difference(polygon2_shapely)
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self.mainwindow, "Warning", "Polygon warning: {}".format(e)
                )
                return

            if isinstance(return_shapely, shapely.Polygon):
                if self.current_graph is None:
                    self.current_graph = Polygon()
                    self.addItem(self.current_graph)

                self.current_graph.hover_alpha = int(
                    self.mainwindow.cfg["software"]["polygon_alpha_hover"] * 255
                )
                self.current_graph.nohover_alpha = int(
                    self.mainwindow.cfg["software"]["polygon_alpha_no_hover"] * 255
                )

                for point in return_shapely.exterior.coords:
                    x, y = point[0], point[1]
                    self.current_graph.addPoint(QtCore.QPointF(x, y))

                self.current_graph.set_drawed(
                    category, group, iscrowd, note, color, layer
                )
                self.mainwindow.polygons.insert(index, self.current_graph)
                self.current_graph = None

            elif isinstance(return_shapely, shapely.MultiPolygon):
                for return_shapely_polygon in return_shapely.geoms:
                    if self.current_graph is None:
                        self.current_graph = Polygon()
                        self.addItem(self.current_graph)

                    self.current_graph.hover_alpha = int(
                        self.mainwindow.cfg["software"]["polygon_alpha_hover"] * 255
                    )
                    self.current_graph.nohover_alpha = int(
                        self.mainwindow.cfg["software"]["polygon_alpha_no_hover"] * 255
                    )

                    for point in return_shapely_polygon.exterior.coords:
                        x, y = point[0], point[1]
                        self.current_graph.addPoint(QtCore.QPointF(x, y))

                    self.current_graph.set_drawed(
                        category, group, iscrowd, note, color, layer
                    )
                    self.mainwindow.polygons.insert(index, self.current_graph)
                    self.current_graph = None

            # 删除旧的多边形
            for polygon_item in self.selected_polygons_list:
                self.mainwindow.polygons.remove(polygon_item)
                polygon_item.delete()
                self.removeItem(polygon_item)
                del polygon_item
            self.selected_polygons_list.clear()

            self.mainwindow.annos_dock_widget.update_listwidget()

    def polygons_intersection(self):
        """Intersect. Only support two polygons. Always use the attributes of the first polygon."""
        if len(self.selected_polygons_list) == 2:
            index = self.mainwindow.polygons.index(self.selected_polygons_list[0])

            category = self.selected_polygons_list[0].category
            group = self.selected_polygons_list[0].group
            iscrowd = self.selected_polygons_list[0].iscrowd
            note = self.selected_polygons_list[0].note
            layer = self.selected_polygons_list[0].zValue()
            color = self.selected_polygons_list[0].color
            try:
                polygon1_shapely = shapely.Polygon(
                    [
                        (point.x(), point.y())
                        for point in self.selected_polygons_list[0].vertices
                    ]
                )
                polygon2_shapely = shapely.Polygon(
                    [
                        (point.x(), point.y())
                        for point in self.selected_polygons_list[1].vertices
                    ]
                )
                return_shapely = polygon1_shapely.intersection(polygon2_shapely)
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self.mainwindow, "Warning", "Polygon warning: {}".format(e)
                )
                return

            if isinstance(return_shapely, shapely.Polygon):
                if self.current_graph is None:
                    self.current_graph = Polygon()
                    self.addItem(self.current_graph)

                self.current_graph.hover_alpha = int(
                    self.mainwindow.cfg["software"]["polygon_alpha_hover"] * 255
                )
                self.current_graph.nohover_alpha = int(
                    self.mainwindow.cfg["software"]["polygon_alpha_no_hover"] * 255
                )

                for point in return_shapely.exterior.coords:
                    x, y = point[0], point[1]
                    self.current_graph.addPoint(QtCore.QPointF(x, y))

                self.current_graph.set_drawed(
                    category, group, iscrowd, note, color, layer
                )
                self.mainwindow.polygons.insert(index, self.current_graph)
                self.current_graph = None

            elif isinstance(return_shapely, shapely.MultiPolygon):
                for return_shapely_polygon in return_shapely.geoms:
                    if self.current_graph is None:
                        self.current_graph = Polygon()
                        self.addItem(self.current_graph)

                    self.current_graph.hover_alpha = int(
                        self.mainwindow.cfg["software"]["polygon_alpha_hover"] * 255
                    )
                    self.current_graph.nohover_alpha = int(
                        self.mainwindow.cfg["software"]["polygon_alpha_no_hover"] * 255
                    )

                    for point in return_shapely_polygon.exterior.coords:
                        x, y = point[0], point[1]
                        self.current_graph.addPoint(QtCore.QPointF(x, y))

                    self.current_graph.set_drawed(
                        category, group, iscrowd, note, color, layer
                    )
                    self.mainwindow.polygons.insert(index, self.current_graph)
                    self.current_graph = None

            # 删除旧的多边形
            for polygon_item in self.selected_polygons_list:
                self.mainwindow.polygons.remove(polygon_item)
                polygon_item.delete()
                self.removeItem(polygon_item)
                del polygon_item
            self.selected_polygons_list.clear()

            self.mainwindow.annos_dock_widget.update_listwidget()

    def polygons_symmetric_difference(self):
        """Exclude. Only support two polygons. Always use the attributes of the first polygon."""
        if len(self.selected_polygons_list) == 2:
            index = self.mainwindow.polygons.index(self.selected_polygons_list[0])

            category = self.selected_polygons_list[0].category
            group = self.selected_polygons_list[0].group
            iscrowd = self.selected_polygons_list[0].iscrowd
            note = self.selected_polygons_list[0].note
            layer = self.selected_polygons_list[0].zValue()
            color = self.selected_polygons_list[0].color
            try:
                polygon1_shapely = shapely.Polygon(
                    [
                        (point.x(), point.y())
                        for point in self.selected_polygons_list[0].vertices
                    ]
                )
                polygon2_shapely = shapely.Polygon(
                    [
                        (point.x(), point.y())
                        for point in self.selected_polygons_list[1].vertices
                    ]
                )
                return_shapely = polygon1_shapely.symmetric_difference(polygon2_shapely)
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self.mainwindow, "Warning", "Polygon warning: {}".format(e)
                )
                return

            if isinstance(return_shapely, shapely.Polygon):
                if self.current_graph is None:
                    self.current_graph = Polygon()
                    self.addItem(self.current_graph)

                self.current_graph.hover_alpha = int(
                    self.mainwindow.cfg["software"]["polygon_alpha_hover"] * 255
                )
                self.current_graph.nohover_alpha = int(
                    self.mainwindow.cfg["software"]["polygon_alpha_no_hover"] * 255
                )

                for point in return_shapely.exterior.coords:
                    x, y = point[0], point[1]
                    self.current_graph.addPoint(QtCore.QPointF(x, y))

                self.current_graph.set_drawed(
                    category, group, iscrowd, note, color, layer
                )
                self.mainwindow.polygons.insert(index, self.current_graph)
                self.current_graph = None

            elif isinstance(return_shapely, shapely.MultiPolygon):
                for return_shapely_polygon in return_shapely.geoms:
                    if self.current_graph is None:
                        self.current_graph = Polygon()
                        self.addItem(self.current_graph)

                    self.current_graph.hover_alpha = int(
                        self.mainwindow.cfg["software"]["polygon_alpha_hover"] * 255
                    )
                    self.current_graph.nohover_alpha = int(
                        self.mainwindow.cfg["software"]["polygon_alpha_no_hover"] * 255
                    )

                    for point in return_shapely_polygon.exterior.coords:
                        x, y = point[0], point[1]
                        self.current_graph.addPoint(QtCore.QPointF(x, y))

                    self.current_graph.set_drawed(
                        category, group, iscrowd, note, color, layer
                    )
                    self.mainwindow.polygons.insert(index, self.current_graph)
                    self.current_graph = None

            # 删除旧的多边形
            for polygon_item in self.selected_polygons_list:
                self.mainwindow.polygons.remove(polygon_item)
                polygon_item.delete()
                self.removeItem(polygon_item)
                del polygon_item
            self.selected_polygons_list.clear()

            self.mainwindow.annos_dock_widget.update_listwidget()

    def mousePressEvent(self, event: "QtWidgets.QGraphicsSceneMouseEvent"):
        pos = event.scenePos()
        if pos.x() < 0:
            pos.setX(0)
        if pos.x() > self.width() - 1:
            pos.setX(self.width() - 1)
        if pos.y() < 0:
            pos.setY(0)
        if pos.y() > self.height() - 1:
            pos.setY(self.height() - 1)
        if self.mode == STATUSMode.CREATE:
            # 拖动鼠标描点
            self.last_draw_time = time.time()
            self.pressed = True

            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                if self.draw_mode == DRAWMode.SEGMENTANYTHING_POINT:
                    self.prompt_point_positions.append([pos.x(), pos.y()])
                    self.prompt_point_labels.append(1)
                    prompt_point_item = PromptPoint(pos, 1)
                    prompt_point_item.setVisible(
                        self.mainwindow.cfg["software"]["show_prompt"]
                    )
                    self.prompt_point_items.append(prompt_point_item)
                    self.addItem(prompt_point_item)

                elif self.draw_mode == DRAWMode.SEGMENTANYTHING_BOX:  # sam 矩形框提示
                    if len(self.prompt_box_item.points) < 1:
                        self.prompt_box_item.addPoint(pos)
                        self.prompt_box_item.addPoint(pos)
                    else:
                        self.finish_draw()

                elif self.draw_mode == DRAWMode.SEGMENTANYTHING_VISUAL:
                    if len(self.prompt_visual_current_item.points) < 1:
                        self.prompt_visual_current_item.addPoint(pos)
                        self.prompt_visual_current_item.addPoint(pos)
                    else:
                        self.prompt_visual_current_item.removePoint(len(self.prompt_visual_current_item.points) - 1)
                        self.prompt_visual_current_item.addPoint(pos)

                        # add to list
                        self.prompt_visual_items.append(self.prompt_visual_current_item)
                        self.prompt_visual_labels.append(self.prompt_visual_current_label)

                        self.prompt_visual_current_item = None
                        self.change_mode_to_view()

                elif self.draw_mode == DRAWMode.POLYGON:
                    # 移除随鼠标移动的点
                    point = self.current_graph.removePoint(len(self.current_graph.points) - 1)
                    if point is not None:
                        pos = point
                    # 添加当前点
                    self.current_graph.addPoint(pos)
                    # 添加随鼠标移动的点
                    self.current_graph.addPoint(pos)
                else:
                    raise ValueError("The draw mode named {} not supported.")
            if event.button() == QtCore.Qt.MouseButton.RightButton:
                if self.draw_mode == DRAWMode.SEGMENTANYTHING_POINT:
                    self.prompt_point_positions.append([pos.x(), pos.y()])
                    self.prompt_point_labels.append(0)
                    prompt_point_item = PromptPoint(pos, 0)
                    prompt_point_item.setVisible(
                        self.mainwindow.cfg["software"]["show_prompt"]
                    )
                    self.prompt_point_items.append(prompt_point_item)
                    self.addItem(prompt_point_item)

                elif self.draw_mode == DRAWMode.POLYGON:
                    pass
                elif self.draw_mode == DRAWMode.SEGMENTANYTHING_BOX:
                    try:
                        self.finish_draw()
                    except:
                        pass
                else:
                    raise ValueError("The draw mode named {} not supported.")
            if self.draw_mode == DRAWMode.SEGMENTANYTHING_POINT:
                self.update_mask()

        if self.mode == STATUSMode.REPAINT:
            # 拖动鼠标描点
            self.last_draw_time = time.time()
            self.pressed = True

            if self.repaint_start_vertex is None:
                # 开始repaint
                if self.hovered_vertex is not None:
                    self.repaint_start_vertex = self.hovered_vertex
                    self.repaint_line_item.addPoint(
                        self.repaint_start_vertex.pos()
                    )  # 添加当前点
                    self.repaint_line_item.addPoint(
                        self.repaint_start_vertex.pos()
                    )  # 添加随鼠标移动的点
            else:
                # 结束repaint
                if (
                    self.hovered_vertex is not None
                    and self.hovered_vertex.polygon == self.repaint_start_vertex.polygon
                ):
                    self.repaint_end_vertex = self.hovered_vertex

                    # 移除随鼠标移动的点
                    self.repaint_line_item.removePoint(len(self.repaint_line_item.points) - 1)
                    # 添加结束点
                    self.repaint_line_item.addPoint(self.repaint_end_vertex.pos())

                    repaint_polygon = self.repaint_start_vertex.polygon
                    repaint_start_index = repaint_polygon.vertices.index(
                        self.repaint_start_vertex
                    )
                    repaint_end_index = repaint_polygon.vertices.index(
                        self.repaint_end_vertex
                    )
                    replace_points = [
                        QtCore.QPointF(vertex.pos())
                        for vertex in self.repaint_line_item.vertices
                    ]

                    if repaint_start_index > repaint_end_index:
                        record = repaint_start_index
                        repaint_start_index = repaint_end_index
                        repaint_end_index = record
                        replace_points = replace_points[::-1]

                    # 这里永远替换数量最少的顶点
                    distance = abs(repaint_end_index - repaint_start_index)
                    if len(repaint_polygon.vertices) - distance < distance:
                        # 替换两端的点
                        points = [
                            vertex.pos()
                            for vertex in repaint_polygon.vertices[
                                repaint_start_index + 1 : repaint_end_index
                            ]
                        ] + replace_points[::-1]
                    else:
                        # 替换中间的点
                        points = (
                            [
                                vertex.pos()
                                for vertex in repaint_polygon.vertices[
                                    :repaint_start_index
                                ]
                            ]
                            + replace_points
                            + [
                                vertex.pos()
                                for vertex in repaint_polygon.vertices[
                                    repaint_end_index + 1 :
                                ]
                            ]
                        )
                    repaint_polygon.delete()
                    for point in points:
                        repaint_polygon.addPoint(point)
                    repaint_polygon.redraw()
                    self.mainwindow.set_saved_state(False)

                    self.repaint_line_item.delete()  # 清除所有路径
                    self.removeItem(self.repaint_line_item)
                    self.repaint_line_item = None

                    self.repaint_start_vertex = None
                    self.repaint_end_vertex = None
                    self.change_mode_to_view()
                else:
                    # 移除随鼠标移动的点
                    self.repaint_line_item.removePoint(len(self.repaint_line_item.points) - 1)
                    # 添加当前点
                    self.repaint_line_item.addPoint(pos)
                    # 添加随鼠标移动的点
                    self.repaint_line_item.addPoint(pos)

        self.mainwindow.plugin_manager_dialog.trigger_on_mouse_press(pos)

        super(AnnotationScene, self).mousePressEvent(event)

    def _constrain_to_angle(self, last_point: QtCore.QPointF, current_point: QtCore.QPointF) -> QtCore.QPointF:
        """
        将当前点约束到与上一个点成0°、45°、90°、135°等角度的位置

        Args:
            last_point: 上一个点的位置
            current_point: 当前鼠标位置

        Returns:
            约束后的点位置
        """
        # 计算相对坐标
        dx = current_point.x() - last_point.x()
        dy = current_point.y() - last_point.y()

        # 计算距离
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < 1:  # 距离太小，直接返回
            return current_point

        # 计算当前角度（弧度）
        angle_rad = math.atan2(dy, dx)
        # 转换为角度
        angle_deg = math.degrees(angle_rad)

        # 约束到最近的45度倍数（0°, 45°, 90°, 135°, 180°, -45°, -90°, -135°）
        # 将角度归一化到[-180, 180]范围
        if angle_deg > 180:
            angle_deg -= 360

        # 找到最近的45度倍数
        constrained_angle_deg = round(angle_deg / 45.0) * 45.0

        # 转换回弧度
        constrained_angle_rad = math.radians(constrained_angle_deg)

        # 根据约束后的角度和距离计算新位置
        new_x = last_point.x() + distance * math.cos(constrained_angle_rad)
        new_y = last_point.y() + distance * math.sin(constrained_angle_rad)

        return QtCore.QPointF(new_x, new_y)

    # 拖动鼠标描点
    def mouseReleaseEvent(self, event: "QtWidgets.QGraphicsSceneMouseEvent"):
        self.pressed = False

        pos = event.scenePos()
        if pos.x() < 0:
            pos.setX(0)
        if pos.x() > self.width() - 1:
            pos.setX(self.width() - 1)
        if pos.y() < 0:
            pos.setY(0)
        if pos.y() > self.height() - 1:
            pos.setY(self.height() - 1)

        self.mainwindow.plugin_manager_dialog.trigger_on_mouse_release(pos)

        super(AnnotationScene, self).mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: "QtWidgets.QGraphicsSceneMouseEvent"):
        # 辅助线
        if self.guide_line_x is not None and self.guide_line_y is not None:
            if self.guide_line_x in self.items():
                self.removeItem(self.guide_line_x)

            if self.guide_line_y in self.items():
                self.removeItem(self.guide_line_y)

            self.guide_line_x = None
            self.guide_line_y = None

        pos = event.scenePos()
        if pos.x() < 0:
            pos.setX(0)
        if pos.x() > self.width() - 1:
            pos.setX(self.width() - 1)
        if pos.y() < 0:
            pos.setY(0)
        if pos.y() > self.height() - 1:
            pos.setY(self.height() - 1)
        # 限制在图片范围内

        if self.mode == STATUSMode.CREATE:
            if self.draw_mode == DRAWMode.POLYGON:
                # 随鼠标位置实时更新多边形
                # 如果按住Shift键，则约束到横平竖直或45度角
                if self.shift_pressed and len(self.current_graph.points) >= 2:
                    # 获取上一个点（倒数第二个点）
                    last_point = self.current_graph.points[-2]
                    # 计算约束后的位置
                    pos = self._constrain_to_angle(last_point, pos)
                self.current_graph.movePoint(len(self.current_graph.points) - 1, pos)

            elif self.draw_mode == DRAWMode.SEGMENTANYTHING_BOX:
                if self.prompt_box_item is not None:
                    self.prompt_box_item.movePoint(len(self.prompt_box_item.points) - 1, pos)
                    self.update_mask()

            elif self.draw_mode == DRAWMode.SEGMENTANYTHING_VISUAL:
                if self.prompt_visual_current_item is not None:
                    self.prompt_visual_current_item.movePoint(len(self.prompt_visual_current_item.points) - 1, pos)

            else:
                pass

        if self.mode == STATUSMode.REPAINT:
            self.repaint_line_item.movePoint(len(self.repaint_line_item.points) - 1, pos)

        pen = QtGui.QPen()
        pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        # 辅助线
        if self.guide_line_x is None and self.width() > 0 and self.height() > 0:
            self.guide_line_x = QtWidgets.QGraphicsLineItem(
                QtCore.QLineF(pos.x(), 0, pos.x(), self.height())
            )
            self.guide_line_x.setPen(pen)
            self.guide_line_x.setZValue(1)
            self.addItem(self.guide_line_x)
        if self.guide_line_y is None and self.width() > 0 and self.height() > 0:
            self.guide_line_y = QtWidgets.QGraphicsLineItem(
                QtCore.QLineF(0, pos.y(), self.width(), pos.y())
            )
            self.guide_line_y.setPen(pen)
            self.guide_line_y.setZValue(1)
            self.addItem(self.guide_line_y)

        # 状态栏,显示当前坐标
        if self.image_data is not None:
            x, y = round(pos.x()), round(pos.y())
            self.mainwindow.labelCoord.setText("xy: ({:>4d},{:>4d})".format(x, y))

            data = self.image_data[y][x]
            if self.image_data.ndim == 2:
                self.mainwindow.labelData.setText("pix: [{:^3d}]".format(data))
            elif self.image_data.ndim == 3:
                if len(data) == 3:
                    self.mainwindow.labelData.setText(
                        "rgb: [{:>3d},{:>3d},{:>3d}]".format(data[0], data[1], data[2])
                    )
                else:
                    self.mainwindow.labelData.setText("pix: [{}]".format(data))

        # 拖动鼠标描点
        if self.pressed:  # 拖动鼠标
            current_time = time.time()
            if (
                self.last_draw_time is not None
                and current_time - self.last_draw_time < self.draw_interval
            ):
                return  # 时间小于给定值不画点
            self.last_draw_time = current_time

            if self.current_graph is not None:
                if self.draw_mode == DRAWMode.POLYGON:
                    # 移除随鼠标移动的点
                    self.current_graph.removePoint(len(self.current_graph.points) - 1)
                    # 添加当前点
                    self.current_graph.addPoint(pos)
                    # 添加随鼠标移动的点
                    self.current_graph.addPoint(pos)

            if self.mode == STATUSMode.REPAINT and self.repaint_line_item is not None:
                # 移除随鼠标移动的点
                self.repaint_line_item.removePoint(len(self.repaint_line_item.points) - 1)
                # 添加当前点
                self.repaint_line_item.addPoint(pos)
                # 添加随鼠标移动的点
                self.repaint_line_item.addPoint(pos)

            self.mainwindow.plugin_manager_dialog.trigger_on_mouse_pressed_and_mouse_move(
                pos
            )

        self.mainwindow.plugin_manager_dialog.trigger_on_mouse_move(pos)

        super(AnnotationScene, self).mouseMoveEvent(event)

    def update_mask(self):
        """Update the mask output of sam."""
        if not self.mainwindow.use_segment_anything:
            return
        if self.image_data is None:
            return
        if not (self.image_data.ndim == 3 and self.image_data.shape[-1] == 3):
            return

        if len(self.prompt_point_positions) > 0 and len(self.prompt_point_labels) > 0:
            mask = self.mainwindow.segany.predict_with_point_prompt(
                self.prompt_point_positions, self.prompt_point_labels
            )
            self.mask = mask
            color = np.array([0, 0, 255])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            mask_image = mask_image.astype("uint8")
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
            mask_image = cv2.addWeighted(
                self.image_data, self.mask_alpha, mask_image, 1, 0
            )
        elif self.prompt_box_item is not None:
            if len(self.prompt_box_item.points) < 2:
                return
            point1 = self.prompt_box_item.points[0]
            point2 = self.prompt_box_item.points[1]
            box = np.array(
                [
                    min(point1.x(), point2.x()),
                    min(point1.y(), point2.y()),
                    max(point1.x(), point2.x()),
                    max(point1.y(), point2.y()),
                ]
            )
            mask = self.mainwindow.segany.predict_with_box_prompt(box)

            self.mask = mask
            color = np.array([0, 0, 255])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            mask_image = mask_image.astype("uint8")
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
            # 这里通过调整原始图像的权重self.mask_alpha，来调整mask的明显程度。
            mask_image = cv2.addWeighted(
                self.image_data, self.mask_alpha, mask_image, 1, 0
            )
        else:
            mask_image = np.zeros(self.image_data.shape, dtype=np.uint8)
            mask_image = cv2.addWeighted(self.image_data, 1, mask_image, 0, 0)
        mask_image = QtGui.QImage(
            mask_image.tobytes(),
            mask_image.shape[1],
            mask_image.shape[0],
            mask_image.shape[1] * 3,
            QtGui.QImage.Format_RGB888,
        )
        mask_pixmap = QtGui.QPixmap(mask_image)
        if self.mask_item is not None:
            self.mask_item.setPixmap(mask_pixmap)

    def backspace(self):
        """Backspace to the previous annotation state. Only work with create mode."""
        if self.mode == STATUSMode.CREATE:
            # 返回上一步操作
            if self.draw_mode == DRAWMode.SEGMENTANYTHING_POINT:
                if len(self.prompt_point_positions) > 0:
                    self.prompt_point_positions.pop()
                if len(self.prompt_point_labels) > 0:
                    self.prompt_point_labels.pop()
                if len(self.prompt_point_items) > 0:
                    prompt_point_item = self.prompt_point_items.pop()
                    self.removeItem(prompt_point_item)
                    del prompt_point_item
                self.update_mask()
            elif self.draw_mode == DRAWMode.POLYGON:
                if len(self.current_graph.points) < 2:
                    return
                # 移除随鼠标移动的点
                self.current_graph.removePoint(len(self.current_graph.points) - 2)

        if self.mode == STATUSMode.REPAINT:
            if len(self.repaint_line_item.points) < 2:
                return
            self.repaint_line_item.removePoint(len(self.repaint_line_item.points) - 2)


class AnnotationView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super(AnnotationView, self).__init__(parent)
        self.setMouseTracking(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)

        self.ctrl_pressed = False
        self.shift_pressed = False
        self.factor = 1.2
        self.scroll = 40

        # 影响了窗口截图功能，暂时注释掉
        # self.setViewport(QtWidgets.QOpenGLWidget())
        # self.setRenderHint(QtGui.QPainter.Antialiasing, False)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Control:
            self.ctrl_pressed = True
            if self.scene():  # 同步到scene
                self.scene().ctrl_pressed = True
        if event.key() == QtCore.Qt.Key.Key_Shift:
            self.shift_pressed = True
            if self.scene():  # 同步到scene
                self.scene().shift_pressed = True
        super(AnnotationView, self).keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Control:
            self.ctrl_pressed = False
            if self.scene():  # 同步到scene
                self.scene().ctrl_pressed = False
        if event.key() == QtCore.Qt.Key.Key_Shift:
            self.shift_pressed = False
            if self.scene():  # 同步到scene
                self.scene().shift_pressed = False
        super(AnnotationView, self).keyReleaseEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent):
        angel = event.angleDelta()
        angelX, angelY = angel.x(), angel.y()
        point = event.pos()  # 当前鼠标位置

        if self.shift_pressed:
            self.horizontal_scroll(angelY)

        elif self.ctrl_pressed:
            self.vertical_scroll(angelY)

        else:
            if angelY > 0:
                self.zoom(self.factor, point)
            else:
                self.zoom(1 / self.factor, point)

    def zoom_in(self):
        self.zoom(self.factor)

    def zoom_out(self):
        self.zoom(1 / self.factor)

    def zoomfit(self):
        self.fitInView(
            0,
            0,
            self.scene().width(),
            self.scene().height(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
        )

    def zoom(self, factor, point=None):
        mouse_old = self.mapToScene(point) if point is not None else None
        # 缩放比例

        pix_widget = (
            self.transform()
            .scale(factor, factor)
            .mapRect(QtCore.QRectF(0, 0, 1, 1))
            .width()
        )
        if pix_widget > 30 and factor > 1:
            return
        if pix_widget < 0.01 and factor < 1:
            return

        self.scale(factor, factor)
        if point is not None:
            mouse_now = self.mapToScene(point)
            center_now = self.mapToScene(
                self.viewport().width() // 2, self.viewport().height() // 2
            )
            center_new = mouse_old - mouse_now + center_now
            self.centerOn(center_new)

    def horizontal_scroll(self, angle):
        scroll_amount = angle / 120 * self.scroll
        h_scroll = self.horizontalScrollBar()
        h_scroll.setValue(h_scroll.value() - int(scroll_amount))

    def vertical_scroll(self, angle):
        scroll_amount = angle / 120 * self.scroll
        v_scroll = self.verticalScrollBar()
        v_scroll.setValue(v_scroll.value() - int(scroll_amount))