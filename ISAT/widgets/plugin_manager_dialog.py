# -*- coding: utf-8 -*-
# @Author  : LG
import os
import importlib
import inspect
from ISAT.ui.plugin_manager_dialog import Ui_Dialog
from ISAT.widgets.plugin_base import PluginBase
from PyQt5 import QtWidgets, QtCore, QtGui
import sys


class PluginManagerDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent, mainwindow):
        super(PluginManagerDialog, self).__init__(parent)
        self.mainwindow = mainwindow
        self.setupUi(self)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.Stretch)

        self.plugin_dir = 'ISAT/plugins'
        self.plugins = []

        self.pushButton_close.clicked.connect(self.close)
        self.load_plugins()

    def load_plugins(self):
        self.tableWidget.setRowCount(0)
        """从插件目录加载所有插件"""
        print('loading plugins')
        if not os.path.exists(self.plugin_dir):
            os.makedirs(self.plugin_dir)
            return

        self.tableWidget.setColumnWidth(0, 100)
        self.tableWidget.setColumnWidth(1, 250)
        self.tableWidget.setColumnWidth(2, 150)
        self.tableWidget.setColumnWidth(3, 150)

        for plugin_name in os.listdir(self.plugin_dir):
            plugin_path = os.path.join(self.plugin_dir, plugin_name)
            if os.path.isdir(plugin_path):
                self._load_single_plugin(plugin_path, plugin_name)

        self.update_gui()

    def _load_single_plugin(self, plugin_path, plugin_name):
        """加载单个插件"""
        init_path = os.path.join(plugin_path, "main.py")
        module_name = f"plugins.{plugin_name}"

        if os.path.exists(init_path):
            try:
                # 动态导入插件模块
                spec = importlib.util.spec_from_file_location(module_name, init_path)
                plugin_module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = plugin_module
                spec.loader.exec_module(plugin_module)

                # 查找插件实现类
                for name, obj in inspect.getmembers(plugin_module):
                    if inspect.isclass(obj) and issubclass(obj, PluginBase) and obj != PluginBase:
                        plugin_instance = obj()
                        self._initialize_plugin(plugin_instance)

            except Exception as e:
                print(f"Failed to load plugin {plugin_name}: {str(e)}")

    def _initialize_plugin(self, plugin_instance):
        """初始化插件"""
        try:
            plugin_instance.init_plugin(self.mainwindow)
            self.plugins.append(plugin_instance)
            print(f"Loaded plugin: {plugin_instance.get_plugin_name()}")
        except Exception as e:
            print(f"Failed to initialize plugin: {str(e)}")

    def update_gui(self):
        self.tableWidget.setRowCount(0)
        row = 0
        for plugin_instance in self.plugins:
            activate_checkbox = QtWidgets.QCheckBox()
            activate_checkbox.stateChanged.connect(plugin_instance.activate_state_changed)
            plugin_name_item = QtWidgets.QTableWidgetItem(plugin_instance.get_plugin_name())
            plugin_author_item = QtWidgets.QTableWidgetItem(plugin_instance.get_plugin_author())
            plugin_author_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            plugin_version_item = QtWidgets.QTableWidgetItem(plugin_instance.get_plugin_version())
            plugin_version_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            plugin_description_item = QtWidgets.QTableWidgetItem(plugin_instance.get_plugin_description())

            self.tableWidget.insertRow(self.tableWidget.rowCount())
            self.tableWidget.setCellWidget(row, 0, activate_checkbox)
            self.tableWidget.setItem(row, 1, plugin_name_item)
            self.tableWidget.setItem(row, 2, QtWidgets.QTableWidgetItem(plugin_author_item))
            self.tableWidget.setItem(row, 3, QtWidgets.QTableWidgetItem(plugin_version_item))
            self.tableWidget.setItem(row, 4, QtWidgets.QTableWidgetItem(plugin_description_item))

            row += 1

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

    def trigger_after_sam_encode_finished(self, index):
        for plugin in self.plugins:
            if plugin.enabled:
                plugin.after_sam_encode_finished_event(index)

    def trigger_on_mouse_move(self, scene_pos):
        for plugin in self.plugins:
            if plugin.enabled:
                plugin.on_mouse_move_event(scene_pos)

    def trigger_on_mouse_release(self, scene_pos):
        for plugin in self.plugins:
            if plugin.enabled:
                plugin.on_mouse_release_event(scene_pos)

    def trigger_on_mouse_press(self, scene_pos):
        for plugin in self.plugins:
            if plugin.enabled:
                plugin.on_mouse_press_event(scene_pos)

    def trigger_on_mouse_pressed_and_mouse_move(self, scene_pos):
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