# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtWidgets, QtGui, QtCore
from enum import Enum
from widgets.polygon import Polygon
from configs import STATUSMode, CLICKMode, DRAWMode
from PIL import Image
import numpy as np
import cv2


class AnnotationScene(QtWidgets.QGraphicsScene):
    def __init__(self, mainwindow):
        super(AnnotationScene, self).__init__()
        self.mainwindow = mainwindow
        self.image_item:QtWidgets.QGraphicsPixmapItem = None
        self.image_data = None
        self.current_graph:Polygon = None
        self.mode = STATUSMode.VIEW
        self.click = CLICKMode.POSITIVE
        self.draw_mode = DRAWMode.SEGMENTANYTHING   # 默认使用segment anything进行快速标注
        self.click_points = []
        self.click_points_mode = []
        self.masks:np.ndarray = None
        self.top_layer = 1

        self.guide_line_x:QtWidgets.QGraphicsLineItem = None
        self.guide_line_y:QtWidgets.QGraphicsLineItem = None

    def load_image(self, image_path:str):
        self.clear()

        self.image_data = np.array(Image.open(image_path))
        if self.mainwindow.use_segment_anything:
            self.mainwindow.segany.reset_image()

            if self.image_data.ndim == 3 and self.image_data.shape[-1] == 3:
                self.mainwindow.segany.set_image(self.image_data)
            elif self.image_data.ndim == 2 and image_path.endswith('.png'):
                # 单通道图标签图
                pass
            else:
                QtWidgets.QMessageBox.warning(self.mainwindow, 'Warning', 'Segment anything only support 3 channel rgb image.')

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

    def change_mode_to_view(self):
        self.mode = STATUSMode.VIEW
        self.image_item.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.ArrowCursor))

        self.mainwindow.actionPrev.setEnabled(True)
        self.mainwindow.actionNext.setEnabled(True)

        self.mainwindow.actionSegment_anything.setEnabled(self.mainwindow.use_segment_anything)
        self.mainwindow.actionPolygon.setEnabled(True)
        self.mainwindow.actionBackspace.setEnabled(False)
        self.mainwindow.actionFinish.setEnabled(False)
        self.mainwindow.actionCancel.setEnabled(False)

        self.mainwindow.actionTo_top.setEnabled(False)
        self.mainwindow.actionTo_bottom.setEnabled(False)
        self.mainwindow.actionEdit.setEnabled(False)
        self.mainwindow.actionDelete.setEnabled(False)
        self.mainwindow.actionSave.setEnabled(True)

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

        if self.draw_mode == DRAWMode.SEGMENTANYTHING:
            # mask to polygon
            # --------------
            if self.masks is not None:
                masks = self.masks
                masks = masks.astype('uint8') * 255
                h, w = masks.shape[-2:]
                masks = masks.reshape(h, w)

                contours, _ = cv2.findContours(masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

                # 这里取轮廓点数最多的（可能返回多个轮廓）
                contour = contours[0]
                for cont in contours:
                    if len(cont) > len(contour):
                        contour = cont

                for point in contour:
                    x, y = point[0]
                    self.current_graph.addPoint(QtCore.QPointF(x, y))

        elif self.draw_mode == DRAWMode.POLYGON:
            if len(self.current_graph.points) < 1:
                return

            # 移除鼠标移动点
            # self.current_graph.removePoint(len(self.current_graph.points) - 1)

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

        # 选择类别
        self.mainwindow.category_choice_widget.load_cfg()
        self.mainwindow.category_choice_widget.show()

        # mask清空
        self.click_points.clear()
        self.click_points_mode.clear()
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
        self.update_mask()

    def delete_selected_graph(self):
        deleted_layer = None
        for item in self.selectedItems():
            if item in self.mainwindow.polygons:
                self.mainwindow.polygons.remove(item)
                item.delete()
                self.removeItem(item)
                deleted_layer = item.zValue()
                del item
        if deleted_layer is not None:
            for p in self.mainwindow.polygons:
                if p.zValue() > deleted_layer:
                    p.setZValue(p.zValue() - 1)
            self.mainwindow.labels_dock_widget.update_listwidget()

    def edit_polygon(self):
        selectd_items = self.selectedItems()
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
            sceneX = self.width() if sceneX > self.width() else sceneX
            sceneY = 0 if sceneY < 0 else sceneY
            sceneY = self.height() if sceneY > self.height() else sceneY

            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                if self.draw_mode == DRAWMode.SEGMENTANYTHING:
                    self.click_points.append([sceneX, sceneY])
                    self.click_points_mode.append(1)
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
                elif self.draw_mode == DRAWMode.POLYGON:
                    pass
                else:
                    raise ValueError('The draw mode named {} not supported.')
            if self.draw_mode == DRAWMode.SEGMENTANYTHING:
                self.update_mask()
        super(AnnotationScene, self).mousePressEvent(event)

    def mouseMoveEvent(self, event: 'QtWidgets.QGraphicsSceneMouseEvent'):
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
        if pos.x() > self.width(): pos.setX(self.width())
        if pos.y() < 0: pos.setY(0)
        if pos.y() > self.height(): pos.setY(self.height())
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

            data = self.image_data[y-1][x-1]
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
        if not (self.image_data.ndim == 3 and self.image_data.shape[-1] == 3):
            return

        if len(self.click_points) > 0 and len(self.click_points_mode) > 0:
            masks = self.mainwindow.segany.predict(self.click_points, self.click_points_mode)
            self.masks = masks
            color = np.array([0, 0, 255])
            h, w = masks.shape[-2:]
            mask_image = masks.reshape(h, w, 1) * color.reshape(1, 1, -1)
            mask_image = mask_image.astype("uint8")
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
            mask_image = cv2.addWeighted(self.image_data, 0.5, mask_image, 0.9, 0)
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
        if pix_widget > 3 or pix_widget < 0.01:
            return

        self.scale(factor, factor)
        if point is not None:
            mouse_now = self.mapToScene(point)
            center_now = self.mapToScene(self.viewport().width() // 2, self.viewport().height() // 2)
            center_new = mouse_old - mouse_now + center_now
            self.centerOn(center_new)
