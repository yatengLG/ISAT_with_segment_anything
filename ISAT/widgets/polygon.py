# -*- coding: utf-8 -*-
# @Author  : LG

import math
import typing

from PyQt5 import QtCore, QtGui, QtWidgets

from ISAT.annotation import Object
from ISAT.configs import STATUSMode, ShapeType


# ============================================================
#  Prompt point (SAM point prompt visual)
# ============================================================

class PromptPoint(QtWidgets.QGraphicsPathItem):
    """SAM prompt point."""

    def __init__(self, pos, type=0):
        super(PromptPoint, self).__init__()
        self.color = QtGui.QColor("#0000FF") if type == 0 else QtGui.QColor("#00FF00")
        self.color.setAlpha(255)
        self.painterpath = QtGui.QPainterPath()
        self.painterpath.addEllipse(QtCore.QRectF(-1, -1, 2, 2))
        self.setPath(self.painterpath)
        self.setBrush(self.color)
        self.setPen(QtGui.QPen(self.color, 3))
        self.setZValue(1e5)

        self.setPos(pos)


# ============================================================
#  Vertex hierarchy
# ============================================================

class BaseVertex(QtWidgets.QGraphicsPathItem):
    """Base class for draggable handle vertices: common appearance,
    scene-boundary clamping and selection highlighting."""

    def __init__(self, parent_shape, color, nohover_size=2, selectable=True):
        super().__init__()
        self.parent_shape = parent_shape
        self.color = QtGui.QColor(color)
        self.color.setAlpha(255)
        self.nohover_size = nohover_size
        self.hover_size = self.nohover_size + 2
        self.line_width = 0

        self.nohover_path = self._make_ellipse(self.nohover_size)
        self.hover_path = self._make_ellipse(self.hover_size)

        self.setPath(self.nohover_path)
        self.setBrush(self.color)
        self.setPen(QtGui.QPen(self.color, self.line_width))
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, selectable)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(1e5)

    @staticmethod
    def _make_ellipse(size):
        """Create a circle-shaped QPainterPath centred at origin."""
        path = QtGui.QPainterPath()
        path.addEllipse(QtCore.QRectF(-size // 2, -size // 2, size, size))
        return path

    def setColor(self, color):
        """Update the vertex colour."""
        self.color = QtGui.QColor(color)
        self.color.setAlpha(255)
        self.setPen(QtGui.QPen(self.color, self.line_width))
        self.setBrush(self.color)

    def itemChange(
        self, change: "QtWidgets.QGraphicsItem.GraphicsItemChange", value: typing.Any
    ):
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self.scene().mainwindow.actionDelete.setEnabled(self.isSelected())
            if self.isSelected():
                self.setBrush(QtGui.QColor("#00A0FF"))
            else:
                self.color.setAlpha(255)
                self.setBrush(self.color)

        if (
            change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionChange
            and self.isEnabled()
        ):
            value = self._clamp_to_scene(value)
            index = self.parent_shape.vertices.index(self)
            self.parent_shape.movePoint(index, value)

        return super().itemChange(change, value)

    def _clamp_to_scene(self, value):
        """Constrain the vertex position to lie within the scene bounds."""
        if value.x() < 0:
            value.setX(0)
        if value.x() > self.scene().width() - 1:
            value.setX(self.scene().width() - 1)
        if value.y() < 0:
            value.setY(0)
        if value.y() > self.scene().height() - 1:
            value.setY(self.scene().height() - 1)
        return value


class PolygonVertex(BaseVertex):
    """Vertex for polygon annotation — selectable with hover effects."""

    def __init__(self, parent_shape, color, nohover_size=2):
        super().__init__(parent_shape, color, nohover_size, selectable=True)

    def hoverEnterEvent(self, event: "QGraphicsSceneHoverEvent"):
        self.scene().hovered_vertex = self
        if self.scene().mode == STATUSMode.CREATE:
            self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
        else:  # EDIT, VIEW
            self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.OpenHandCursor))
            if not self.isSelected():
                self.setBrush(QtGui.QColor(255, 255, 255, 255))
            self.setPath(self.hover_path)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: "QGraphicsSceneHoverEvent"):
        self.scene().hovered_vertex = None
        if not self.isSelected():
            self.color.setAlpha(255)
            self.setBrush(self.color)
        self.setPath(self.nohover_path)
        super().hoverLeaveEvent(event)


class LineVertex(BaseVertex):
    """Vertex for repaint guide line — non-selectable."""

    def __init__(self, parent_shape, color, nohover_size=2):
        super().__init__(parent_shape, color, nohover_size, selectable=False)


class PromptRectVertex(BaseVertex):
    """Vertex for SAM prompt rectangle — non-selectable."""

    def __init__(self, parent_shape, color, nohover_size=2):
        super().__init__(parent_shape, color, nohover_size, selectable=False)


class OBBVertex(PolygonVertex):
    """Vertex for OBB annotation — selectable with hover effects.

    Currently identical to PolygonVertex; exists as a dedicated type so that
    OBB-specific vertex behaviour (e.g. rotation handle) can be added later
    without affecting Polygon vertices.
    """

    def __init__(self, parent_shape, color, nohover_size=2):
        super().__init__(parent_shape, color, nohover_size)


# ============================================================
#  Shape mixin — points / vertices CRUD shared by Polygon, Line, PromptRect
# ============================================================

class BaseShape:
    """Mixin providing point/vertex management for shapes.

    Subclasses must call ``_init_shape(vertex_cls)`` in ``__init__`` and
    implement ``redraw()`` to re-render the shape from ``self.points``.
    """

    def _init_shape(self, vertex_cls):
        """Initialise the point list and the vertex factory."""
        self.points: list = []
        self.vertices: list = []
        self._vertex_cls = vertex_cls

    def addPoint(self, point: QtCore.QPointF):
        """Append a point and its corresponding visual vertex to the scene."""
        self.points.append(point)
        vertex_size = self.scene().mainwindow.cfg["software"]["vertex_size"] * 2
        vertex = self._vertex_cls(self, self.color, vertex_size)
        self.scene().addItem(vertex)
        self.vertices.append(vertex)
        vertex.setPos(point)

    def _add_trailing(self, point: QtCore.QPointF):
        """Add a mouse-following trailing point.

        Default behaviour equals :meth:`addPoint`; shapes whose ``addPoint``
        carries side effects (e.g. OBB completion) override this method.
        """
        self.addPoint(point)

    def movePoint(self, index: int, point: QtCore.QPointF):
        """Move the *index*-th point to a new scene position."""
        if not 0 <= index < len(self.points):
            return
        self.points[index] = self.mapFromScene(point)
        self.redraw()
        self._on_point_moved(index, point)

    def _on_point_moved(self, index: int, point: QtCore.QPointF):
        """Hook after ``movePoint``; subclasses may override (no-op by default)."""

    def removePoint(self, index):
        """Remove the *index*-th point and its vertex.  Returns the removed point."""
        if not self.points:
            return None
        point = self.points.pop(index)
        vertex = self.vertices.pop(index)
        self.scene().removeItem(vertex)
        del vertex
        self.redraw()
        return point

    def delete(self):
        """Remove all points and vertices (e.g. when discarding the shape)."""
        self.points.clear()
        while self.vertices:
            vertex = self.vertices.pop()
            self.scene().removeItem(vertex)
            del vertex


# ============================================================
#  Polygon — full annotation shape
# ============================================================

class Polygon(QtWidgets.QGraphicsPolygonItem, BaseShape):
    """Polygon annotation shape.

    Key attributes: points/vertices (geometry), is_drawing (drawing state),
    category/group/iscrowd/note/area (annotation data).
    """

    def __init__(self):
        QtWidgets.QGraphicsPolygonItem.__init__(self, parent=None)
        self._init_shape(PolygonVertex)

        self.line_width = 1
        self.hover_alpha = 150
        self.nohover_alpha = 80
        self.category = ""
        self.group = 0
        self.iscrowd = False
        self.note = ""
        self.area = 0

        self.color = QtGui.QColor("#ff0000")
        self.is_drawing = True
        pen = QtGui.QPen(self.color, self.line_width)
        pen.setStyle(QtCore.Qt.PenStyle.DotLine)
        self.setPen(pen)
        self.setBrush(QtGui.QBrush(self.color, QtCore.Qt.BrushStyle.FDiagPattern))

        self.setAcceptHoverEvents(True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True
        )
        self.setZValue(1e5)

    def _on_point_moved(self, index: int, point: QtCore.QPointF):
        """Polygon-specific side-effects after a vertex is dragged."""
        if self.scene().mainwindow.cfg["software"]["real_time_area"]:
            self.area = self.calculate_area()
        if (
            self.scene().mainwindow.load_finished
            and not self.is_drawing
            and self.scene().mode != STATUSMode.REPAINT
        ):
            self.scene().mainwindow.set_saved_state(False)

    def moveVertex(self, index, point):
        """Set a vertex position directly (bypasses movePoint)."""
        if not 0 <= index < len(self.vertices):
            return
        vertex = self.vertices[index]
        vertex.setEnabled(False)
        vertex.setPos(point)
        vertex.setEnabled(True)

    def itemChange(
        self, change: "QGraphicsItem.GraphicsItemChange", value: typing.Any
    ):
        if (
            change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged
            and not self.is_drawing
            and self.scene().mode != STATUSMode.CREATE
        ):
            if self.isSelected():
                color = QtGui.QColor("#00A0FF")
                color.setAlpha(self.hover_alpha)
                self.setBrush(color)
                self.scene().selected_polygons_list.append(self)
            else:
                self.color.setAlpha(self.nohover_alpha)
                self.setBrush(self.color)
                if self in self.scene().selected_polygons_list:
                    self.scene().selected_polygons_list.remove(self)
            self.scene().mainwindow.annos_dock_widget.set_selected(self)

        if (
            change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionChange
        ):
            if self.is_drawing:
                value = 0
            else:
                bias = value
                l, t, b, r = (
                    self.boundingRect().left(),
                    self.boundingRect().top(),
                    self.boundingRect().bottom(),
                    self.boundingRect().right(),
                )
                if l + bias.x() < 0:
                    bias.setX(-l)
                if r + bias.x() > self.scene().width() - 1:
                    bias.setX(self.scene().width() - 1 - r)
                if t + bias.y() < 0:
                    bias.setY(-t)
                if b + bias.y() > self.scene().height() - 1:
                    bias.setY(self.scene().height() - 1 - b)

                for index, point in enumerate(self.points):
                    self.moveVertex(index, point + bias)

                if self.scene().mainwindow.load_finished:
                    self.scene().mainwindow.set_saved_state(False)

        if (
            change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged
            and self.isSelected()
        ):
            self.setSelected(not self.is_drawing)
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event: "QGraphicsSceneHoverEvent"):
        if not self.is_drawing and not self.isSelected():
            self.color.setAlpha(self.hover_alpha)
            self.setBrush(self.color)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: "QGraphicsSceneHoverEvent"):
        if not self.is_drawing and not self.isSelected():
            self.color.setAlpha(self.nohover_alpha)
            self.setBrush(self.color)
        super().hoverLeaveEvent(event)

    def mouseDoubleClickEvent(self, event: "QGraphicsSceneMouseEvent"):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.scene().mainwindow.category_edit_widget.polygons = [self]
            self.scene().mainwindow.category_edit_widget.load_cfg()
            self.scene().mainwindow.category_edit_widget.show()

    def redraw(self):
        if len(self.points) < 1:
            return
        self.setPolygon(QtGui.QPolygonF(self.points))

    def change_color(self, color: QtGui.QColor):
        self.color = color
        if not self.scene().mainwindow.cfg["software"]["show_edge"]:
            color.setAlpha(0)
        self.setPen(QtGui.QPen(color, self.line_width))
        self.color.setAlpha(self.nohover_alpha)
        self.setBrush(self.color)

        vertex_color = self.color
        vertex_color.setAlpha(255)
        for vertex in self.vertices:
            vertex.setPen(QtGui.QPen(vertex_color, self.line_width))
            vertex.setBrush(vertex_color)

    def set_drawed(
        self,
        category: str,
        group: int,
        iscrowd: bool,
        note: str,
        color: QtGui.QColor,
        layer: int = None,
    ):
        """Set annotation attributes and mark the shape as finished (is_drawing=False)."""
        self.is_drawing = False
        self.category = category
        if isinstance(group, str):
            group = 0 if group == "" else int(group)
        self.group = group
        self.iscrowd = iscrowd
        self.note = note

        self.color = color
        self.color.setAlpha(255)

        if not self.scene().mainwindow.cfg["software"]["show_edge"]:
            self.color.setAlpha(0)
        self.setPen(QtGui.QPen(self.color, self.line_width))
        self.color.setAlpha(self.nohover_alpha)
        self.setBrush(self.color)
        if layer is not None:
            self.setZValue(layer)
            for vertex in self.vertices:
                vertex.setZValue(layer)
        for vertex in self.vertices:
            vertex.setColor(color)

        self.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
            not self.scene().mainwindow.annos_dock_widget.checkBox_lock.isChecked(),
        )

    def calculate_area(self) -> float:
        """Calculate area of polygon using the shoelace formula."""
        area = 0
        num_points = len(self.points)
        for i in range(num_points):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % num_points]
            d = p1.x() * p2.y() - p2.x() * p1.y()
            area += d
        return abs(area) / 2

    def load_object(self, obj):
        """Load attributes from an Annotation Object."""
        segmentation = obj.segmentation
        for x, y in segmentation:
            point = QtCore.QPointF(x, y)
            self.addPoint(point)
        color = self.scene().mainwindow.category_color_dict.get(obj.category, "#6F737A")
        self.set_drawed(
            obj.category,
            obj.group,
            obj.iscrowd,
            obj.note,
            QtGui.QColor(color),
            obj.layer,
        )
        self.area = obj.area

    def to_object(self) -> Object:
        """Convert to an Annotation Object for serialisation."""
        if self.is_drawing:
            return None
        segmentation = []
        for point in self.points:
            point = point + self.pos()
            segmentation.append((round(point.x(), 2), round(point.y(), 2)))
        xmin = self.boundingRect().x() + self.pos().x()
        ymin = self.boundingRect().y() + self.pos().y()
        xmax = xmin + self.boundingRect().width()
        ymax = ymin + self.boundingRect().height()

        if (
            not self.scene().mainwindow.cfg["software"]["real_time_area"]
            or self.area == 0
        ):
            self.area = self.calculate_area()

        object = Object(
            self.category,
            group=self.group,
            segmentation=segmentation,
            area=self.area,
            layer=self.zValue(),
            bbox=(xmin, ymin, xmax, ymax),
            iscrowd=self.iscrowd,
            note=self.note,
            shape_type=ShapeType.POLYGON.value,
        )
        return object

# ============================================================
#  OBB — Oriented Bounding Box annotation shape
# ============================================================

class OBB(QtWidgets.QGraphicsPolygonItem, BaseShape):
    """Oriented Bounding Box — a rectangle constrained to 4 corners.

    ``self.points`` holds the corners in **local** coordinates, clockwise
    order ``[P0, P1, P3, P2]``: P0→P1 is the first edge, P1→P3 the
    perpendicular (width) edge.  angle / centre / size are **derived** from
    the points so the geometry always stays consistent.

    Creation is driven by the canvas (3 clicks): P0, P1 define the first
    edge; the 3rd click is projected onto the perpendicular through P1 by
    :meth:`_complete_rectangle`, then ``finish_draw`` is called immediately.
    """

    def __init__(self):
        QtWidgets.QGraphicsPolygonItem.__init__(self, parent=None)
        self._init_shape(OBBVertex)

        self.line_width = 1
        self.hover_alpha = 150
        self.nohover_alpha = 80
        self.category = ""
        self.group = 0
        self.iscrowd = False
        self.note = ""
        self.area = 0
        self._out_of_bounds = False  # 出图状态，用于边框高亮提示

        self.color = QtGui.QColor("#ff0000")
        self.is_drawing = True
        pen = QtGui.QPen(self.color, self.line_width)
        pen.setStyle(QtCore.Qt.PenStyle.DotLine)
        self.setPen(pen)
        self.setBrush(QtGui.QBrush(self.color, QtCore.Qt.BrushStyle.FDiagPattern))

        self.setAcceptHoverEvents(True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True
        )
        self.setZValue(1e5)

    @property
    def angle(self) -> float:
        """Rotation angle of the first edge in radians (range [-π, π])."""
        if len(self.points) < 2:
            return 0.0
        d = self.points[1] - self.points[0]
        return math.atan2(d.y(), d.x())

    @property
    def center(self) -> QtCore.QPointF:
        """Centre point of the rectangle (local coordinates)."""
        if len(self.points) < 4:
            return QtCore.QPointF(0, 0)
        return (self.points[0] + self.points[2]) / 2

    @property
    def size(self):
        """Return ``(width, height)`` = lengths of the first and perpendicular edges."""
        if len(self.points) < 4:
            return (0, 0)
        w = math.hypot(
            self.points[1].x() - self.points[0].x(),
            self.points[1].y() - self.points[0].y(),
        )
        h = math.hypot(
            self.points[3].x() - self.points[0].x(),
            self.points[3].y() - self.points[0].y(),
        )
        return (w, h)

    def addPoint(self, point: QtCore.QPointF):
        """Append a corner point.

        Loading from disk allows up to 4 points; interactive drawing blocks
        beyond 3 — the rectangle completion is done by the canvas via
        :meth:`_complete_rectangle`.
        """
        _loading = getattr(self, "_loading", False)
        max_points = 4 if _loading else 3
        if len(self.points) >= max_points:
            return

        super().addPoint(point)

    def _add_trailing(self, point: QtCore.QPointF):
        """Add a mouse-following trailing point (bypasses ``addPoint``)."""
        # pylint: disable=protected-access
        if len(self.points) >= 3:
            return
        super(OBB, self).addPoint(point)

    def _complete_rectangle(self) -> bool:
        """Complete ``[P0, P1, P_click]`` → ``[P0, P1, P3, P2]`` (clockwise).

        Projects P_click onto the perpendicular through P1.  Returns False
        (leaving the points untouched) when the first edge is degenerate.
        """
        p0 = self.points[0]
        p1 = self.points[1]
        p_click = self.points[2]  # 3rd corner, needs projection

        edge = p1 - p0
        # perpendicular: rotate edge by +90°
        d_perp = QtCore.QPointF(-edge.y(), edge.x())

        # Project p_click onto the perpendicular line from P1
        v = p_click - p1
        denom = d_perp.x() * d_perp.x() + d_perp.y() * d_perp.y()
        if denom < 1e-12:
            # Zero-length first edge: perpendicular undefined → no rectangle.
            return False
        t = (v.x() * d_perp.x() + v.y() * d_perp.y()) / denom

        p3 = QtCore.QPointF(p1.x() + t * d_perp.x(), p1.y() + t * d_perp.y())
        p2 = p3 - edge  # == P0 + t * d_perp

        # Replace the clicked corner with its projected position (P3)
        self.points[2] = p3
        self.vertices[2].setPos(p3)

        # Append the 4th corner (P2)
        self.points.append(p2)
        vertex_size = self.scene().mainwindow.cfg["software"]["vertex_size"] * 2
        vertex = self._vertex_cls(self, self.color, vertex_size)
        self.scene().addItem(vertex)
        self.vertices.append(vertex)
        vertex.setPos(p2)
        return True

    def movePoint(self, index: int, point: QtCore.QPointF):
        """Move a corner; the opposite corner stays fixed, angle preserved."""
        if not 0 <= index < len(self.points):
            return
        if len(self.points) < 4:
            super().movePoint(index, point)
            return

        new_corner = self.mapFromScene(point)
        opposite_idx = (index + 2) % 4
        fixed_corner = self.points[opposite_idx]

        self._recompute_from_diagonal(index, new_corner, opposite_idx, fixed_corner)
        self.redraw()
        self._check_bounds()
        self._on_point_moved(index, point)

    # ------------------------------------------------------------------
    #  出图检测：移动 / 旋转 / 拖顶点后检查 OBB 是否超出图像，出图则高亮边框
    # ------------------------------------------------------------------

    def _check_bounds(self):
        """Update the out-of-image state and highlight the border if needed.

        Called after whole-shape move, rotation or vertex drag; restores the
        normal border once the shape is back inside the image.
        """
        if len(self.points) < 4 or self.scene() is None:
            return
        w = self.scene().width()
        h = self.scene().height()
        pos = self.pos()
        out = False
        for p in self.points:
            x, y = p.x() + pos.x(), p.y() + pos.y()
            if x < 0 or x > w - 1 or y < 0 or y > h - 1:
                out = True
                break
        if out != self._out_of_bounds:
            self._out_of_bounds = out
            self._apply_bounds_pen()

    def _apply_bounds_pen(self):
        """Set the border pen according to the current out-of-image state."""
        if self._out_of_bounds:
            # 出图：红色高亮 + 加粗长虚线边框
            pen = QtGui.QPen(QtGui.QColor("#FF0000"), self.line_width + 2)
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        else:
            # 与 Polygon 一致：实线、不透明（show_edge 关闭时隐藏）
            edge_color = QtGui.QColor(self.color)
            edge_color.setAlpha(255)
            if not self.scene().mainwindow.cfg["software"]["show_edge"]:
                edge_color.setAlpha(0)
            pen = QtGui.QPen(edge_color, self.line_width)
        self.setPen(pen)

    def _recompute_from_diagonal(self, dragged_idx, new_pos, fixed_idx, fixed_pos):
        """Recompute all 4 corners from a new diagonal, preserving angle."""
        ang = self.angle
        d1 = QtCore.QPointF(math.cos(ang), math.sin(ang))   # edge direction
        d2 = QtCore.QPointF(-math.sin(ang), math.cos(ang))   # perpendicular

        center = (new_pos + fixed_pos) / 2
        half_diag = new_pos - center  # = (new_pos - fixed_pos) / 2

        hw = half_diag.x() * d1.x() + half_diag.y() * d1.y()  # dot(d1, half_diag)
        hh = half_diag.x() * d2.x() + half_diag.y() * d2.y()  # dot(d2, half_diag)

        self.points[dragged_idx] = new_pos
        self.points[fixed_idx] = fixed_pos
        self.points[(dragged_idx + 1) % 4] = QtCore.QPointF(
            center.x() - d1.x() * hw + d2.x() * hh,
            center.y() - d1.y() * hw + d2.y() * hh,
        )
        self.points[(dragged_idx + 3) % 4] = QtCore.QPointF(
            center.x() + d1.x() * hw - d2.x() * hh,
            center.y() + d1.y() * hw - d2.y() * hh,
        )

        # Sync vertex scene positions
        for i in range(4):
            if i != dragged_idx:
                self.moveVertex(i, self.mapToScene(self.points[i]))

    def rotate(self, delta_angle: float):
        """Rotate the OBB by *delta_angle* radians around its centre."""
        if len(self.points) < 4:
            return

        c = self.center
        cos_a = math.cos(delta_angle)
        sin_a = math.sin(delta_angle)

        for i in range(4):
            dx = self.points[i].x() - c.x()
            dy = self.points[i].y() - c.y()
            self.points[i] = QtCore.QPointF(
                c.x() + dx * cos_a - dy * sin_a,
                c.y() + dx * sin_a + dy * cos_a,
            )
            self.moveVertex(i, self.mapToScene(self.points[i]))

        self.redraw()
        self._check_bounds()

    def moveVertex(self, index, point):
        """Direct vertex position update (bypasses ``movePoint``)."""
        if not 0 <= index < len(self.vertices):
            return
        vertex = self.vertices[index]
        vertex.setEnabled(False)
        vertex.setPos(point)
        vertex.setEnabled(True)

    def _on_point_moved(self, index: int, point: QtCore.QPointF):
        if self.scene().mainwindow.cfg["software"]["real_time_area"]:
            self.area = self.calculate_area()
        if (
            self.scene().mainwindow.load_finished
            and not self.is_drawing
            and self.scene().mode != STATUSMode.REPAINT
        ):
            self.scene().mainwindow.set_saved_state(False)

    def itemChange(
        self, change: "QGraphicsItem.GraphicsItemChange", value: typing.Any
    ):
        if (
            change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged
            and not self.is_drawing
            and self.scene().mode != STATUSMode.CREATE
        ):
            if self.isSelected():
                color = QtGui.QColor("#00A0FF")
                color.setAlpha(self.hover_alpha)
                self.setBrush(color)
                self.scene().selected_polygons_list.append(self)
            else:
                self.color.setAlpha(self.nohover_alpha)
                self.setBrush(self.color)
                if self in self.scene().selected_polygons_list:
                    self.scene().selected_polygons_list.remove(self)
            self.scene().mainwindow.annos_dock_widget.set_selected(self)

        if (
            change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionChange
        ):
            if self.is_drawing:
                value = QtCore.QPointF(0, 0)
            else:
                bias = value
                l, t, b, r = (
                    self.boundingRect().left(),
                    self.boundingRect().top(),
                    self.boundingRect().bottom(),
                    self.boundingRect().right(),
                )
                if l + bias.x() < 0:
                    bias.setX(-l)
                if r + bias.x() > self.scene().width() - 1:
                    bias.setX(self.scene().width() - 1 - r)
                if t + bias.y() < 0:
                    bias.setY(-t)
                if b + bias.y() > self.scene().height() - 1:
                    bias.setY(self.scene().height() - 1 - b)

                for index, point in enumerate(self.points):
                    self.moveVertex(index, point + bias)

                self._check_bounds()
                if self.scene().mainwindow.load_finished:
                    self.scene().mainwindow.set_saved_state(False)

        if (
            change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged
            and self.isSelected()
        ):
            self.setSelected(not self.is_drawing)

        return super().itemChange(change, value)

    def hoverEnterEvent(self, event: "QGraphicsSceneHoverEvent"):
        if not self.is_drawing and not self.isSelected():
            self.color.setAlpha(self.hover_alpha)
            self.setBrush(self.color)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: "QGraphicsSceneHoverEvent"):
        if not self.is_drawing and not self.isSelected():
            self.color.setAlpha(self.nohover_alpha)
            self.setBrush(self.color)
        super().hoverLeaveEvent(event)

    def mouseDoubleClickEvent(self, event: "QGraphicsSceneMouseEvent"):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.scene().mainwindow.category_edit_widget.polygons = [self]
            self.scene().mainwindow.category_edit_widget.load_cfg()
            self.scene().mainwindow.category_edit_widget.show()

    def redraw(self):
        if len(self.points) < 1:
            return
        self.setPolygon(QtGui.QPolygonF(self.points))

    def change_color(self, color: QtGui.QColor):
        self.color = color
        if not self.scene().mainwindow.cfg["software"]["show_edge"]:
            color.setAlpha(0)
        self.setPen(QtGui.QPen(color, self.line_width))
        self.color.setAlpha(self.nohover_alpha)
        self.setBrush(self.color)

        vertex_color = QtGui.QColor(self.color)
        vertex_color.setAlpha(255)
        for vertex in self.vertices:
            vertex.setPen(QtGui.QPen(vertex_color, self.line_width))
            vertex.setBrush(vertex_color)

        self._apply_bounds_pen()

    def set_drawed(
        self,
        category: str,
        group: int,
        iscrowd: bool,
        note: str,
        color: QtGui.QColor,
        layer: int = None,
    ):
        """Mark the OBB as finished and set annotation attributes.

        Enforces the completed-OBB invariant: exactly 4 corners.  A 3-point
        shape is completed via :meth:`_complete_rectangle`; extra points are
        truncated.  Drawing-state shapes never reach this method.
        """
        # 强制：完成的 OBB 必须恰好 4 个顶点（绘制中的 OBB 不会走到这里）。
        if len(self.points) == 3:
            self._complete_rectangle()
            self.redraw()
        elif len(self.points) > 4:
            while len(self.points) > 4:
                self.removePoint(len(self.points) - 1)

        self.is_drawing = False
        self.category = category
        if isinstance(group, str):
            group = 0 if group == "" else int(group)
        self.group = group
        self.iscrowd = iscrowd
        self.note = note

        self.color = QtGui.QColor(color)
        self.color.setAlpha(255)

        if not self.scene().mainwindow.cfg["software"]["show_edge"]:
            self.color.setAlpha(0)
        self.setPen(QtGui.QPen(self.color, self.line_width))
        self.color.setAlpha(self.nohover_alpha)
        self.setBrush(self.color)
        if layer is not None:
            self.setZValue(layer)
            for vertex in self.vertices:
                vertex.setZValue(layer)
        for vertex in self.vertices:
            vertex.setColor(color)

        self.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
            not self.scene().mainwindow.annos_dock_widget.checkBox_lock.isChecked(),
        )

        self._apply_bounds_pen()

    def calculate_area(self) -> float:
        """Return width × height."""
        w, h = self.size
        return w * h

    def load_object(self, obj):
        """Load attributes from an Annotation Object."""
        self._loading = True
        for x, y in obj.segmentation:
            self.addPoint(QtCore.QPointF(x, y))
        self._loading = False

        # 3 点 / 多余点由 set_drawed 强制补全/截断为恰好 4 个顶点
        color = self.scene().mainwindow.category_color_dict.get(
            obj.category, "#6F737A"
        )
        self.set_drawed(
            obj.category,
            obj.group,
            obj.iscrowd,
            obj.note,
            QtGui.QColor(color),
            obj.layer,
        )
        self.area = obj.area

    def _ordered_points(self):
        """Return the 4 corners in a canonical order for serialisation.

        Convention: **clockwise** in image coordinates (y axis points down),
        starting from the top-left-most vertex (minimum x, tie-broken by
        minimum y).  Internal ``self.points`` keeps its own (drag-rotated)
        order — only the exported point list is normalised, so the saved
        ``segmentation`` order is stable regardless of editing history and
        directly usable by YOLO-OBB / DOTA-style consumers.
        """
        pts = list(self.points)
        if len(pts) != 4:
            return pts
        start = min(range(4), key=lambda i: (pts[i].x(), pts[i].y()))
        cw = [pts[(start + k) % 4] for k in range(4)]
        ccw = [pts[(start - k) % 4] for k in range(4)]

        def signed_area(seq):
            return 0.5 * sum(
                seq[i].x() * seq[(i + 1) % 4].y()
                - seq[(i + 1) % 4].x() * seq[i].y()
                for i in range(4)
            )

        # Positive shoelace area = clockwise in image coords (y down).
        return cw if signed_area(cw) >= 0 else ccw

    def to_object(self) -> Object:
        """Convert to an Annotation Object for serialisation."""
        if self.is_drawing:
            return None

        # An OBB must serialise as exactly 4 corners — an invalid shape must
        # not reach disk (load_object would silently re-complete it).
        if len(self.points) != 4:
            print(
                "Warning: skip saving invalid OBB with {} points "
                "(exactly 4 required).".format(len(self.points))
            )
            return None

        # 规范化点序：顺时针（图像 y 向下）+ 首点左上（x 最小，并列 y 最小）
        segmentation = []
        for point in self._ordered_points():
            pt = point + self.pos()
            segmentation.append((round(pt.x(), 2), round(pt.y(), 2)))

        xmin = self.boundingRect().x() + self.pos().x()
        ymin = self.boundingRect().y() + self.pos().y()
        xmax = xmin + self.boundingRect().width()
        ymax = ymin + self.boundingRect().height()

        if (
            not self.scene().mainwindow.cfg["software"]["real_time_area"]
            or self.area == 0
        ):
            self.area = self.calculate_area()

        obj = Object(
            self.category,
            group=self.group,
            segmentation=segmentation,
            area=self.area,
            layer=self.zValue(),
            bbox=(xmin, ymin, xmax, ymax),
            iscrowd=self.iscrowd,
            note=self.note,
            shape_type=ShapeType.OBB.value,
        )
        return obj

# ============================================================
#  Line — repaint-mode guide line
# ============================================================

class Line(QtWidgets.QGraphicsPathItem, BaseShape):
    """Visual guide line shown during repaint mode."""

    def __init__(self):
        QtWidgets.QGraphicsPathItem.__init__(self, parent=None)
        self._init_shape(LineVertex)

        self.line_width = 1
        self.color = QtGui.QColor("#ff0000")
        pen = QtGui.QPen(self.color, self.line_width)
        pen.setStyle(QtCore.Qt.PenStyle.DotLine)
        self.setPen(pen)
        self.setZValue(1e5)

    def redraw(self):
        if len(self.points) < 1:
            return

        line_path = QtGui.QPainterPath()
        if self.points:
            line_path.moveTo(self.points[0])
            for point in self.points[1:]:
                line_path.lineTo(point)

        self.setPath(line_path)


# ============================================================
#  PromptRect — SAM box-prompt rectangle
# ============================================================

class PromptRect(QtWidgets.QGraphicsRectItem, BaseShape):
    """Visual rectangle for SAM box-prompt mode."""

    def __init__(self):
        QtWidgets.QGraphicsRectItem.__init__(self, parent=None)
        self._init_shape(PromptRectVertex)

        self.line_width = 1
        self.color = QtGui.QColor("#ff0000")

        pen = QtGui.QPen(self.color, self.line_width)
        pen.setStyle(QtCore.Qt.PenStyle.DotLine)
        self.setPen(pen)

    def redraw(self):
        if len(self.points) < 2:
            return

        self.setRect(QtCore.QRectF(self.points[0], self.points[-1]))
