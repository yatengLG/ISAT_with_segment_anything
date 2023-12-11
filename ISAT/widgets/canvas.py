# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtWidgets, QtGui, QtCore
from ISAT.widgets.polygon import Polygon, Vertex, PromptPoint
from ISAT.configs import STATUSMode, CLICKMode, DRAWMode, CONTOURMode
from PIL import Image
import numpy as np
import cv2
import time # 拖动鼠标描点

class AnnotationScene(QtWidgets.QGraphicsScene):
    def __init__(self, mainwindow):
        super(AnnotationScene, self).__init__()
        self.mainwindow = mainwindow
        self.image_item:QtWidgets.QGraphicsPixmapItem = None
        self.image_data = None
        self.current_graph:Polygon = None
        self.mode = STATUSMode.VIEW
        self.click = CLICKMode.POSITIVE
        self.draw_mode = DRAWMode.SEGMENTANYTHING           # 默认使用segment anything进行快速标注
        self.contour_mode = CONTOURMode.SAVE_EXTERNAL       # 默认SAM只保留外轮廓
        self.click_points = []                              # SAM point prompt
        self.click_points_mode = []                         # SAM point prompt
        self.prompt_points = []
        self.masks:np.ndarray = None
        self.mask_alpha = 0.5
        self.top_layer = 1

        self.guide_line_x:QtWidgets.QGraphicsLineItem = None
        self.guide_line_y:QtWidgets.QGraphicsLineItem = None

        # 拖动鼠标描点     
        self.last_draw_time = time.time()
        self.draw_interval = 0.15
        self.pressd = False

    def load_image(self, image_path:str):
        self.clear()
        if self.mainwindow.use_segment_anything:
            self.mainwindow.segany.reset_image()

        self.image_data = np.array(Image.open(image_path))
                
        self.image_item = QtWidgets.QGraphicsPixmapItem()
        self.image_item.setZValue(0)
        self.addItem(self.image_item)
        self.mask_item = QtWidgets.QGraphicsPixmapItem()
        self.mask_item.setZValue(1)
        self.addItem(self.mask_item)

        self.image_item.setPixmap(QtGui.QPixmap(image_path))
        self.setSceneRect(self.image_item.boundingRect())
        self.change_mode_to_view()

    def change_mode_to_create(self):
        if self.image_item is None:
            return
        self.mode = STATUSMode.CREATE
        self.image_item.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
        self.mainwindow.actionPrev.setEnabled(False)
        self.mainwindow.actionNext.setEnabled(False)

        self.mainwindow.actionSegment_anything.setEnabled(False)
        self.mainwindow.actionPolygon.setEnabled(False)
        self.mainwindow.actionBackspace.setEnabled(True)
        self.mainwindow.actionFinish.setEnabled(True)
        self.mainwindow.actionCancel.setEnabled(True)

        self.mainwindow.actionTo_top.setEnabled(False)
        self.mainwindow.actionTo_bottom.setEnabled(False)
        self.mainwindow.actionEdit.setEnabled(False)
        self.mainwindow.actionDelete.setEnabled(False)
        self.mainwindow.actionSave.setEnabled(False)

        self.mainwindow.set_labels_visible(False)
        self.mainwindow.annos_dock_widget.setEnabled(False)

    def change_mode_to_view(self):
        self.mode = STATUSMode.VIEW
        self.image_item.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.ArrowCursor))

        self.mainwindow.actionPrev.setEnabled(True)
        self.mainwindow.actionNext.setEnabled(True)
        self.mainwindow.SeganyEnabled()
        self.mainwindow.actionPolygon.setEnabled(self.mainwindow.can_be_annotated)
        self.mainwindow.actionBackspace.setEnabled(False)
        self.mainwindow.actionFinish.setEnabled(False)
        self.mainwindow.actionCancel.setEnabled(False)

        self.mainwindow.actionTo_top.setEnabled(False)
        self.mainwindow.actionTo_bottom.setEnabled(False)
        self.mainwindow.actionEdit.setEnabled(False)
        self.mainwindow.actionDelete.setEnabled(False)
        self.mainwindow.actionSave.setEnabled(self.mainwindow.can_be_annotated)

        self.mainwindow.set_labels_visible(True)
        self.mainwindow.annos_dock_widget.setEnabled(True)

    def change_mode_to_edit(self):
        self.mode = STATUSMode.EDIT
        self.image_item.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))

        self.mainwindow.actionPrev.setEnabled(False)
        self.mainwindow.actionNext.setEnabled(False)

        self.mainwindow.actionSegment_anything.setEnabled(False)
        self.mainwindow.actionPolygon.setEnabled(False)
        self.mainwindow.actionBackspace.setEnabled(False)
        self.mainwindow.actionFinish.setEnabled(False)
        self.mainwindow.actionCancel.setEnabled(False)

        self.mainwindow.actionTo_top.setEnabled(True)
        self.mainwindow.actionTo_bottom.setEnabled(True)
        self.mainwindow.actionEdit.setEnabled(True)
        self.mainwindow.actionDelete.setEnabled(True)
        self.mainwindow.actionSave.setEnabled(True)

    def change_click_to_positive(self):
        self.click = CLICKMode.POSITIVE

    def change_click_to_negative(self):
        self.click = CLICKMode.NEGATIVE

    def change_contour_mode_to_save_all(self):
        self.contour_mode = CONTOURMode.SAVE_ALL

    def change_contour_mode_to_save_max_only(self):
        self.contour_mode = CONTOURMode.SAVE_MAX_ONLY

    def change_contour_mode_to_save_external(self):
        self.contour_mode = CONTOURMode.SAVE_EXTERNAL

    def start_segment_anything(self):
        self.draw_mode = DRAWMode.SEGMENTANYTHING
        self.start_draw()

    def start_draw_polygon(self):
        self.draw_mode = DRAWMode.POLYGON
        self.start_draw()

    def start_draw(self):
        # 只有view模式时，才能切换create模式
        if self.mode != STATUSMode.VIEW:
            return
        # 否则，切换到绘图模式
        self.change_mode_to_create()
        # 绘图模式
        if self.mode == STATUSMode.CREATE:
            self.current_graph = Polygon()
            self.addItem(self.current_graph)

    def finish_draw(self):

        if self.current_graph is None:
            return

        self.change_mode_to_view()

        category = self.mainwindow.current_category
        group = self.mainwindow.current_group
        is_crowd = False
        note = ''

        if self.draw_mode == DRAWMode.SEGMENTANYTHING:
            # mask to polygon
            # --------------
            if self.masks is not None:
                masks = self.masks
                masks = masks.astype('uint8') * 255
                h, w = masks.shape[-2:]
                masks = masks.reshape(h, w)

                if self.contour_mode == CONTOURMode.SAVE_ALL:
                    # 当保留所有轮廓时，检测所有轮廓，并建立二层等级关系
                    contours, hierarchy = cv2.findContours(masks, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
                else:
                    # 当只保留外轮廓或单个mask时，只检测外轮廓
                    contours, hierarchy = cv2.findContours(masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

                if self.contour_mode == CONTOURMode.SAVE_MAX_ONLY:
                    largest_contour = max(contours, key=cv2.contourArea)    # 只保留面积最大的轮廓
                    contours = [largest_contour]

                for index, contour in enumerate(contours):
                    # polydp
                    epsilon_factor = 0.001
                    epsilon = epsilon_factor * cv2.arcLength(contour, True)
                    contour = cv2.approxPolyDP(contour, epsilon, True)

                    if self.current_graph is None:
                        self.current_graph = Polygon()
                        self.addItem(self.current_graph)

                    if len(contour) < 3:
                        continue
                    for point in contour:
                        x, y = point[0]
                        self.current_graph.addPoint(QtCore.QPointF(x, y))

                    if self.contour_mode == CONTOURMode.SAVE_ALL and hierarchy[0][index][3] != -1:
                        # 保存所有轮廓，且当前轮廓为子轮廓，则自轮廓类别设置为背景
                        category = '__background__'
                        group = 0
                    else:
                        category = self.mainwindow.current_category
                        group = self.mainwindow.current_group

                    self.current_graph.set_drawed(category,
                                                  group,
                                                  is_crowd,
                                                  note,
                                                  QtGui.QColor(self.mainwindow.category_color_dict[category]),
                                                  self.top_layer)

                    # 添加新polygon
                    self.mainwindow.polygons.append(self.current_graph)
                    # 设置为最高图层
                    self.current_graph.setZValue(len(self.mainwindow.polygons))
                    for vertex in self.current_graph.vertexs:
                        vertex.setZValue(len(self.mainwindow.polygons))
                    self.current_graph = None
                if self.mainwindow.group_select_mode == 'auto':
                    self.mainwindow.current_group += 1
                    self.mainwindow.categories_dock_widget.lineEdit_currentGroup.setText(str(self.mainwindow.current_group))
        elif self.draw_mode == DRAWMode.POLYGON:
            if len(self.current_graph.points) < 1:
                return

            # 移除鼠标移动点
            self.current_graph.removePoint(len(self.current_graph.points) - 1)

            # 单点，删除
            if len(self.current_graph.points) < 2:
                self.current_graph.delete()
                self.removeItem(self.current_graph)
                self.change_mode_to_view()
                return

            # 两点，默认矩形
            if len(self.current_graph.points) == 2:
                first_point = self.current_graph.points[0]
                last_point = self.current_graph.points[-1]
                self.current_graph.removePoint(len(self.current_graph.points) - 1)
                self.current_graph.addPoint(QtCore.QPointF(first_point.x(), last_point.y()))
                self.current_graph.addPoint(last_point)
                self.current_graph.addPoint(QtCore.QPointF(last_point.x(), first_point.y()))

            # 设置polygon 属性
            self.current_graph.set_drawed(category,
                                          group,
                                          is_crowd,
                                          note,
                                          QtGui.QColor(self.mainwindow.category_color_dict[category]),
                                          self.top_layer)
            if self.mainwindow.group_select_mode == 'auto':
                self.mainwindow.current_group += 1
                self.mainwindow.categories_dock_widget.lineEdit_currentGroup.setText(str(self.mainwindow.current_group))
            # 添加新polygon
            self.mainwindow.polygons.append(self.current_graph)
            # 设置为最高图层
            self.current_graph.setZValue(len(self.mainwindow.polygons))
            for vertex in self.current_graph.vertexs:
                vertex.setZValue(len(self.mainwindow.polygons))
        # 选择类别
        # self.mainwindow.category_choice_widget.load_cfg()
        # self.mainwindow.category_choice_widget.show()

        self.mainwindow.annos_dock_widget.update_listwidget()

        self.current_graph = None
        self.change_mode_to_view()

        # mask清空
        self.click_points.clear()
        self.click_points_mode.clear()
        for prompt_point in self.prompt_points:
            try:
                self.removeItem(prompt_point)
            finally:
                del prompt_point
        self.prompt_points.clear()
        self.update_mask()

    def cancel_draw(self):
        if self.current_graph is None:
            return
        self.current_graph.delete() # 清除所有路径
        self.removeItem(self.current_graph)

        self.current_graph = None

        self.change_mode_to_view()

        self.click_points.clear()
        self.click_points_mode.clear()
        for prompt_point in self.prompt_points:
            try:
                self.removeItem(prompt_point)
            finally:
                del prompt_point
        self.prompt_points.clear()

        self.update_mask()

    def delete_selected_graph(self):
        deleted_layer = None
        for item in self.selectedItems():
            if isinstance(item, Polygon) and (item in self.mainwindow.polygons):
                self.mainwindow.polygons.remove(item)
                item.delete()
                self.removeItem(item)
                deleted_layer = item.zValue()
                del item
            elif isinstance(item, Vertex):
                polygon = item.polygon
                if polygon.vertexs:
                    index = polygon.vertexs.index(item)
                    item.polygon.removePoint(index)
                else:
                    self.removeItem(item)
                    del item
                # 如果剩余顶点少于三个，删除多边形
                if len(polygon.vertexs) < 3:
                    if polygon in self.mainwindow.polygons:
                        self.mainwindow.polygons.remove(polygon)
                        polygon.delete()
                    if polygon in self.items():
                        self.removeItem(polygon)
                    deleted_layer = polygon.zValue()
                    del polygon

        if deleted_layer is not None:
            for p in self.mainwindow.polygons:
                if p.zValue() > deleted_layer:
                    p.setZValue(p.zValue() - 1)
            self.mainwindow.annos_dock_widget.update_listwidget()

    def edit_polygon(self):
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
        for vertex in current_polygon.vertexs:
            vertex.setZValue(max_layer)
        self.mainwindow.set_saved_state(False)

    def move_polygon_to_bottom(self):
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
            for vertex in current_polygon.vertexs:
                vertex.setZValue(1)
        self.mainwindow.set_saved_state(False)

    def mousePressEvent(self, event: 'QtWidgets.QGraphicsSceneMouseEvent'):
        if self.mode == STATUSMode.CREATE:
            sceneX, sceneY = event.scenePos().x(), event.scenePos().y()
            sceneX = 0 if sceneX < 0 else sceneX
            sceneX = self.width()-1 if sceneX > self.width()-1 else sceneX
            sceneY = 0 if sceneY < 0 else sceneY
            sceneY = self.height()-1 if sceneY > self.height()-1 else sceneY

            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                if self.draw_mode == DRAWMode.SEGMENTANYTHING:
                    self.click_points.append([sceneX, sceneY])
                    self.click_points_mode.append(1)
                    prompt_point = PromptPoint(QtCore.QPointF(sceneX, sceneY), 1)
                    prompt_point.setVisible(self.mainwindow.show_prompt.checked)
                    self.prompt_points.append(prompt_point)
                    self.addItem(prompt_point)

                elif self.draw_mode == DRAWMode.POLYGON:
                    # 移除随鼠标移动的点
                    self.current_graph.removePoint(len(self.current_graph.points) - 1)
                    # 添加当前点
                    self.current_graph.addPoint(QtCore.QPointF(sceneX, sceneY))
                    # 添加随鼠标移动的点
                    self.current_graph.addPoint(QtCore.QPointF(sceneX, sceneY))
                else:
                    raise ValueError('The draw mode named {} not supported.')
            if event.button() == QtCore.Qt.MouseButton.RightButton:
                if self.draw_mode == DRAWMode.SEGMENTANYTHING:
                    self.click_points.append([sceneX, sceneY])
                    self.click_points_mode.append(0)
                    prompt_point = PromptPoint(QtCore.QPointF(sceneX, sceneY), 0)
                    prompt_point.setVisible(self.mainwindow.show_prompt.checked)
                    self.prompt_points.append(prompt_point)
                    self.addItem(prompt_point)

                elif self.draw_mode == DRAWMode.POLYGON:
                    pass
                else:
                    raise ValueError('The draw mode named {} not supported.')
            if self.draw_mode == DRAWMode.SEGMENTANYTHING:
                self.update_mask()
            
            # 拖动鼠标描点
            self.last_draw_time = time.time()
            self.pressd = True
        super(AnnotationScene, self).mousePressEvent(event)

    # 拖动鼠标描点 
    def mouseReleaseEvent(self, event: 'QtWidgets.QGraphicsSceneMouseEvent'):       
        self.pressd = False
        super(AnnotationScene, self).mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: 'QtWidgets.QGraphicsSceneMouseEvent'):
        # 拖动鼠标描点 
        if self.pressd: # 拖动鼠标
            current_time = time.time()
            if self.last_draw_time is not None and current_time - self.last_draw_time < self.draw_interval:
                return  # 时间小于给定值不画点
            self.last_draw_time = current_time
            sceneX, sceneY = event.scenePos().x(), event.scenePos().y()
            sceneX = 0 if sceneX < 0 else sceneX
            sceneX = self.width()-1 if sceneX > self.width()-1 else sceneX
            sceneY = 0 if sceneY < 0 else sceneY
            sceneY = self.height()-1 if sceneY > self.height()-1 else sceneY
            
            if self.current_graph is not None:
                if self.draw_mode == DRAWMode.POLYGON:
                    # 移除随鼠标移动的点
                    self.current_graph.removePoint(len(self.current_graph.points) - 1)
                    # 添加当前点
                    self.current_graph.addPoint(QtCore.QPointF(sceneX, sceneY))
                    # 添加随鼠标移动的点
                    self.current_graph.addPoint(QtCore.QPointF(sceneX, sceneY))

        # 辅助线
        if self.guide_line_x is not None and self.guide_line_y is not None:
            if self.guide_line_x in self.items():
                self.removeItem(self.guide_line_x)

            if self.guide_line_y in self.items():
                self.removeItem(self.guide_line_y)

            self.guide_line_x = None
            self.guide_line_y = None

        pos = event.scenePos()
        if pos.x() < 0: pos.setX(0)
        if pos.x() > self.width()-1: pos.setX(self.width()-1)
        if pos.y() < 0: pos.setY(0)
        if pos.y() > self.height()-1: pos.setY(self.height()-1)
        # 限制在图片范围内

        if self.mode == STATUSMode.CREATE:
            if self.draw_mode == DRAWMode.POLYGON:
                # 随鼠标位置实时更新多边形
                self.current_graph.movePoint(len(self.current_graph.points)-1, pos)

        # 辅助线
        if self.guide_line_x is None and self.width()>0 and self.height()>0:
            self.guide_line_x = QtWidgets.QGraphicsLineItem(QtCore.QLineF(pos.x(), 0, pos.x(), self.height()))
            self.guide_line_x.setZValue(1)
            self.addItem(self.guide_line_x)
        if self.guide_line_y is None and self.width()>0 and self.height()>0:
            self.guide_line_y = QtWidgets.QGraphicsLineItem(QtCore.QLineF(0, pos.y(), self.width(), pos.y()))
            self.guide_line_y.setZValue(1)
            self.addItem(self.guide_line_y)

        # 状态栏,显示当前坐标
        if self.image_data is not None:
            x, y = round(pos.x()), round(pos.y())
            self.mainwindow.labelCoord.setText('xy: ({:>4d},{:>4d})'.format(x, y))

            data = self.image_data[y][x]
            if self.image_data.ndim == 2:
                self.mainwindow.labelData.setText('pix: [{:^3d}]'.format(data))
            elif self.image_data.ndim == 3:
                if len(data) == 3:
                    self.mainwindow.labelData.setText('rgb: [{:>3d},{:>3d},{:>3d}]'.format(data[0], data[1], data[2]))
                else:
                    self.mainwindow.labelData.setText('pix: [{}]'.format(data))

        super(AnnotationScene, self).mouseMoveEvent(event)

    def update_mask(self):
        if not self.mainwindow.use_segment_anything:
            return
        if self.image_data is None:
            return
        if not (self.image_data.ndim == 3 and self.image_data.shape[-1] == 3):
            return

        if len(self.click_points) > 0 and len(self.click_points_mode) > 0:
            masks = self.mainwindow.segany.predict_with_point_prompt(self.click_points, self.click_points_mode)
            self.masks = masks
            color = np.array([0, 0, 255])
            h, w = masks.shape[-2:]
            mask_image = masks.reshape(h, w, 1) * color.reshape(1, 1, -1)
            mask_image = mask_image.astype("uint8")
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
            # 这里通过调整原始图像的权重self.mask_alpha，来调整mask的明显程度。
            mask_image = cv2.addWeighted(self.image_data, self.mask_alpha, mask_image, 1, 0)
            mask_image = QtGui.QImage(mask_image[:], mask_image.shape[1], mask_image.shape[0], mask_image.shape[1] * 3,
                                      QtGui.QImage.Format_RGB888)
            mask_pixmap = QtGui.QPixmap(mask_image)
            self.mask_item.setPixmap(mask_pixmap)
        else:
            mask_image = np.zeros(self.image_data.shape, dtype=np.uint8)
            mask_image = cv2.addWeighted(self.image_data, 1, mask_image, 0, 0)
            mask_image = QtGui.QImage(mask_image[:], mask_image.shape[1], mask_image.shape[0], mask_image.shape[1] * 3,
                                      QtGui.QImage.Format_RGB888)
            mask_pixmap = QtGui.QPixmap(mask_image)
            self.mask_item.setPixmap(mask_pixmap)

    def backspace(self):
        if self.mode != STATUSMode.CREATE:
            return
        # 返回上一步操作
        if self.draw_mode == DRAWMode.SEGMENTANYTHING:
            if len(self.click_points) > 0:
                self.click_points.pop()
            if len(self.click_points_mode) > 0:
                self.click_points_mode.pop()
            if len(self.prompt_points) > 0:
                prompt_point = self.prompt_points.pop()
                self.removeItem(prompt_point)
                del prompt_point
            self.update_mask()
        elif self.draw_mode == DRAWMode.POLYGON:
            if len(self.current_graph.points) < 2:
                return
            # 移除随鼠标移动的点
            self.current_graph.removePoint(len(self.current_graph.points) - 2)


class AnnotationView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super(AnnotationView, self).__init__(parent)
        self.setMouseTracking(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.factor = 1.2

    def wheelEvent(self, event: QtGui.QWheelEvent):
        angel = event.angleDelta()
        angelX, angelY = angel.x(), angel.y()
        point = event.pos() # 当前鼠标位置
        if angelY > 0:
            self.zoom(self.factor, point)
        else:
            self.zoom(1 / self.factor, point)

    def zoom_in(self):
        self.zoom(self.factor)

    def zoom_out(self):
        self.zoom(1/self.factor)

    def zoomfit(self):
        self.fitInView(0, 0, self.scene().width(), self.scene().height(),  QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def zoom(self, factor, point=None):
        mouse_old = self.mapToScene(point) if point is not None else None
        # 缩放比例

        pix_widget = self.transform().scale(factor, factor).mapRect(QtCore.QRectF(0, 0, 1, 1)).width()
        if pix_widget > 30 and factor > 1: return
        if pix_widget < 0.01 and factor < 1: return

        self.scale(factor, factor)
        if point is not None:
            mouse_now = self.mapToScene(point)
            center_now = self.mapToScene(self.viewport().width() // 2, self.viewport().height() // 2)
            center_new = mouse_old - mouse_now + center_now
            self.centerOn(center_new)
