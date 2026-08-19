# -*- coding: utf-8 -*-
# @Author  : LG

import sys
from importlib.metadata import entry_points

from PyQt5 import QtCore


class PluginManager:
    """Plugin engine: discovery, lifecycle and event dispatch.

    UI-free — this class only keeps the loaded plugin instances and fires
    the lifecycle events to the enabled ones.  The view
    (``PluginManagerDialog``) is responsible for presentation only, so the
    event dispatch never depends on the dialog being shown.
    """

    def __init__(self, mainwindow):
        self.mainwindow = mainwindow
        self.plugins = []

    def load_plugins(self):
        for plugin in self.plugins:
            plugin.disable_plugin()
        self.plugins = []

        print("loading plugins")
        if sys.version_info >= (3, 10):
            eps = entry_points().select(group="isat.plugins")
        else:
            eps = entry_points().get("isat.plugins", [])
        for ep in eps:
            try:
                plugin_class = ep.load()
                plugin_instance = plugin_class()
                plugin_instance.init_plugin(self.mainwindow)
                self.plugins.append(plugin_instance)
                print("loaded plugin: ", plugin_instance.get_plugin_name())
            except Exception as e:
                print("failed to load plugin [{ep}]: ", e)

    # ------------------------------------------------------------------
    #  Lifecycle / event dispatch — iterate enabled plugins only.
    # ------------------------------------------------------------------

    def trigger_before_image_open(self, image_path):
        for plugin in self.plugins:
            if plugin.enabled:
                plugin.before_image_open_event(image_path)

    def trigger_after_image_open(self):
        for plugin in self.plugins:
            if plugin.enabled:
                plugin.after_image_open_event()

    def trigger_before_annotation_start(self):
        for plugin in self.plugins:
            if plugin.enabled:
                plugin.before_annotation_start_event()

    def trigger_after_annotation_created(self):
        for plugin in self.plugins:
            if plugin.enabled:
                plugin.after_annotation_created_event()

    def trigger_after_annotation_changed(self):
        for plugin in self.plugins:
            if plugin.enabled:
                plugin.after_annotation_changed_event()

    def trigger_before_annotations_save(self):
        for plugin in self.plugins:
            if plugin.enabled:
                plugin.before_annotations_save_event()

    def trigger_after_annotations_saved(self):
        for plugin in self.plugins:
            if plugin.enabled:
                plugin.after_annotations_saved_event()

    def trigger_after_sam_encode_finished(self, index: int):
        for plugin in self.plugins:
            if plugin.enabled:
                plugin.after_sam_encode_finished_event(index)

    def trigger_on_mouse_move(self, scene_pos: QtCore.QPointF):
        for plugin in self.plugins:
            if plugin.enabled:
                plugin.on_mouse_move_event(scene_pos)

    def trigger_on_mouse_release(self, scene_pos: QtCore.QPointF):
        for plugin in self.plugins:
            if plugin.enabled:
                plugin.on_mouse_release_event(scene_pos)

    def trigger_on_mouse_press(self, scene_pos: QtCore.QPointF):
        for plugin in self.plugins:
            if plugin.enabled:
                plugin.on_mouse_press_event(scene_pos)

    def trigger_on_mouse_pressed_and_mouse_move(self, scene_pos: QtCore.QPointF):
        for plugin in self.plugins:
            if plugin.enabled:
                plugin.on_mouse_pressed_and_mouse_move_event(scene_pos)

    def trigger_application_start(self):
        for plugin in self.plugins:
            if plugin.enabled:
                plugin.application_start_event()

    def trigger_application_shutdown(self):
        for plugin in self.plugins:
            if plugin.enabled:
                plugin.application_shutdown_event()
