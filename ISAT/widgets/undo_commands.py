# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtWidgets

from ISAT.widgets.polygon import OBB, Polygon


def snapshot_shape(shape):
    """Serialise one finished shape into an ``Object`` (scene coordinates).

    ``to_object()`` bakes the item offset into every point, so the snapshot
    is a faithful scene-space description that ``load_object()`` can rebuild.
    Returns None for shapes that are still being drawn.
    """
    if getattr(shape, "is_drawing", False):
        return None
    return shape.to_object()


def _snapshots_match(before, after):
    """True when two serialised Objects describe the same annotation."""
    if before.category != after.category or before.group != after.group:
        return False
    if before.shape_type != after.shape_type:
        return False
    if before.note != after.note or before.iscrowd != after.iscrowd:
        return False
    return before.segmentation == after.segmentation


def make_shape(scene, obj):
    """Create and add a fresh visual shape from a serialised Object."""
    shape_cls = OBB if getattr(obj, "shape_type", "polygon") == "obb" else Polygon
    shape = shape_cls()
    scene.addItem(shape)
    shape.load_object(obj)
    return shape


def insert_shape_at(scene, index, obj):
    """Insert a rebuilt shape at *index* without touching its neighbours."""
    mainwindow = scene.mainwindow
    new_shape = make_shape(scene, obj)
    insert_at = min(index, len(mainwindow.polygons))
    mainwindow.polygons.insert(insert_at, new_shape)
    mainwindow.annos_dock_widget.update_listwidget()
    return new_shape


def pop_shape_at(scene, index):
    """Remove the shape occupying *index* (no-op when the slot is empty)."""
    mainwindow = scene.mainwindow
    if not 0 <= index < len(mainwindow.polygons):
        mainwindow.annos_dock_widget.update_listwidget()
        return None
    current = mainwindow.polygons.pop(index)
    if current in scene.selected_polygons_list:
        scene.selected_polygons_list.remove(current)
    # 必须在 removeItem 之前清选中：Polygon.itemChange 会访问 self.scene()
    if current.isSelected():
        current.setSelected(False)
    current.delete()
    if current in scene.items():
        scene.removeItem(current)
    mainwindow.annos_dock_widget.update_listwidget()
    return current


def replace_shape_at(scene, index, new_obj):
    """Replace the shape currently at *index* with *new_obj*.

    Undo/redo operate by **list position**, never by a saved object
    reference: multiple commands may act on the same logical shape, and each
    undo rebuilds the shape, so a reference captured earlier would point at
    an already-deleted item.  Looking the current shape up by ``index``
    keeps the chain consistent across any number of stacked commands.

    Returns the freshly created shape, or None when ``new_obj`` is None
    (the slot is simply emptied).
    """
    mainwindow = scene.mainwindow

    # remove whatever currently occupies this slot
    if 0 <= index < len(mainwindow.polygons):
        current = mainwindow.polygons.pop(index)
        if current in scene.selected_polygons_list:
            scene.selected_polygons_list.remove(current)
        # 先清选中（此时 item 仍在 scene 中），再删除
        if current.isSelected():
            current.setSelected(False)
        current.delete()
        if current in scene.items():
            scene.removeItem(current)

    if new_obj is None:
        mainwindow.annos_dock_widget.update_listwidget()
        return None

    new_shape = make_shape(scene, new_obj)
    insert_at = min(index, len(mainwindow.polygons))
    mainwindow.polygons.insert(insert_at, new_shape)
    mainwindow.annos_dock_widget.update_listwidget()
    return new_shape


def snapshot_all_shapes(scene):
    """Ordered snapshot (== z order) of every finished shape on the scene.

    Used by operations that touch several shapes at once (multi-delete, z
    reshuffle, boolean ops): the whole list is captured so undo can restore
    both the shapes and their stacking order.
    """
    snapshot = []
    for item in scene.mainwindow.polygons:
        obj = snapshot_shape(item)
        if obj is not None:
            snapshot.append(obj)
    return snapshot


def _scene_snapshots_match(before, after):
    """True when two whole-scene snapshots describe the same annotations."""
    if len(before) != len(after):
        return False
    for obj_a, obj_b in zip(before, after):
        if not _snapshots_match(obj_a, obj_b):
            return False
        if obj_a.layer != obj_b.layer:
            return False
    return True


def restore_all_shapes(scene, snapshot):
    """Rebuild the entire annotation scene from an ordered snapshot.

    Every finished shape is torn down and re-created in snapshot order, so
    shape geometry, attributes *and* z/stacking order all come back exactly.
    Prompt/mask items and the drawing state are untouched.
    """
    mainwindow = scene.mainwindow

    # tear down (reverse order keeps removals O(1)-ish and avoids churn)
    while mainwindow.polygons:
        shape = mainwindow.polygons.pop()
        if shape in scene.selected_polygons_list:
            scene.selected_polygons_list.remove(shape)
        # 先清选中（item 仍在 scene 中）再删除，避免 itemChange 访问空 scene
        if shape.isSelected():
            shape.setSelected(False)
        shape.delete()
        if shape in scene.items():
            scene.removeItem(shape)

    for obj in snapshot:
        shape = make_shape(scene, obj)
        mainwindow.polygons.append(shape)

    mainwindow.annos_dock_widget.update_listwidget()
    scene.clear_selection_ui()


class ShapeStateCommand(QtWidgets.QUndoCommand):
    """Undoable change of a single finished shape slot.

    Holds the shape's position *index* in ``mainwindow.polygons`` plus the
    serialised ``Object`` before/after the change.  The command is created
    *after* the edit already happened, so the first ``redo()`` (invoked by
    ``QUndoStack.push``) is a no-op.

    Three flavours are expressed by which snapshot is None:

    * ``before=None``  → **add**: redo inserts the shape, undo pops it
    * ``after=None``   → **delete**: redo pops the shape, undo inserts it
    * both present     → **modify**: redo/undo replace the shape in place

    Add/delete must NOT use "replace the slot" semantics: after an add is
    undone the slot index is occupied by a *different* shape (its neighbour),
    so a replace-on-redo would delete that neighbour.
    """

    def __init__(self, scene, text, shape, before_obj, after_obj):
        super(ShapeStateCommand, self).__init__(text)
        self.scene = scene
        self.index = scene.mainwindow.polygons.index(shape)
        self.before_obj = before_obj
        self.after_obj = after_obj
        self._first_redo = True

    def undo(self):
        self._apply(self.before_obj, going_back=True)
        self._after_apply()

    def redo(self):
        if self._first_redo:
            # 命令构造时场景已处于 after 状态（操作已完成），
            # push 触发的首次 redo 必须真正 no-op，不得重建 item，
            # 否则会在 Qt 释放 mouse grabber 前删除被拖动的对象。
            self._first_redo = False
            return
        self._apply(self.after_obj, going_back=False)
        self._after_apply()

    def _apply(self, target_obj, going_back):
        """Apply one side of the change with the correct slot semantics."""
        if target_obj is None:
            # moving towards "no shape here" -> pop (delete)
            pop_shape_at(self.scene, self.index)
            return
        if (self.before_obj is None) or (self.after_obj is None):
            # add / delete pair: the shape enters or leaves the slot
            if going_back:
                # undo(add) pops; undo(delete) inserts
                if self.before_obj is None:
                    pop_shape_at(self.scene, self.index)
                else:
                    insert_shape_at(self.scene, self.index, target_obj)
            else:
                # redo(delete) pops; redo(add) inserts
                if self.after_obj is None:
                    pop_shape_at(self.scene, self.index)
                else:
                    insert_shape_at(self.scene, self.index, target_obj)
            return
        # modify: replace the slot in place
        replace_shape_at(self.scene, self.index, target_obj)

    def _after_apply(self):
        # ``update_listwidget`` inside replace/remove already marks the
        # annotation as unsaved (or autosaves); no extra call needed here.
        self.scene.clear_selection_ui()


class SceneStateCommand(QtWidgets.QUndoCommand):
    """Undoable change that affects several shapes and/or their stacking order.

    Holds two whole-scene snapshots (ordered lists of serialised ``Object``):
    ``before`` and ``after``.  Used by multi-shape destructive operations —
    deletion (which also renumbers z), z reshuffle, boolean ops — where a
    per-shape command cannot express the result.

    Like :class:`ShapeStateCommand`, the command is created *after* the
    change already happened, so the first ``redo()`` fired by
    ``QUndoStack.push`` is a true no-op.
    """

    def __init__(self, scene, text, before, after):
        super(SceneStateCommand, self).__init__(text)
        self.scene = scene
        self.before = before
        self.after = after
        self._first_redo = True

    def undo(self):
        restore_all_shapes(self.scene, self.before)

    def redo(self):
        if self._first_redo:
            # 场景已处于 after 状态，首次 redo 不重建
            self._first_redo = False
            return
        restore_all_shapes(self.scene, self.after)

