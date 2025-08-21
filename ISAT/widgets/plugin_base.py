# -*- coding: utf-8 -*-
# @Author  : LG

from abc import ABC, abstractmethod


class PluginBase(ABC):
    """Base class for all plugins."""
    def __init__(self):
        self.enabled = False

    @abstractmethod
    def init_plugin(self, mainwindow):
        """
        Be used to plugin initialization.
s
        Arguments:
            mainwindow (MainWindow): Get attributes and functions from mainwindow.
        """
        pass

    @abstractmethod
    def enable_plugin(self):
        """
        Enable plugin
        """
        pass

    @abstractmethod
    def disable_plugin(self):
        """
        Disable plugin
        """
        pass

    @abstractmethod
    def get_plugin_author(self) -> str:
        """
        Get plugin author.

        Returns:
            str: Plugin author.
        """
        pass

    @abstractmethod
    def get_plugin_version(self) -> str:
        """
        Get plugin version.

        Returns:
            str: Plugin version.
        """
        pass

    @abstractmethod
    def get_plugin_description(self) -> str:
        """
        Get plugin description.

        Returns:
            str: Plugin description.
        """
        pass

    def get_plugin_name(self) -> str:
        """
        Get plugin name.

        Returns:
            str: Plugin name. Default is the plugin class name.
        """
        return self.__class__.__name__

    def activate_state_changed(self, checkbox_state):
        if checkbox_state:
            self.enable_plugin()
        else:
            self.disable_plugin()

    def before_image_open_event(self, image_path: str):
        """
        当图片打开前调用

        Arguments:
            image_path (str): The path of the image.
        """
        pass

    def after_image_open_event(self):
        """
        当图片打开后调用
        """
        pass

    def before_annotation_start_event(self):
        """
        当开始标注前调用
        """
        pass

    def after_annotation_created_event(self):
        """
        当标注创建完成后调用
        """
        pass

    def after_annotation_changed_event(self):
        """
        当标注发生变化后调用，包括创建新标注，拖动顶点，移动多边形，改变多边形属性等
        """
        pass

    def before_annotations_save_event(self):
        """
        当保存标注文件前调用
        """
        pass

    def after_annotations_saved_event(self):
        """
        当保存标注文件后调用
        """
        pass

    def after_sam_encode_finished_event(self, index):
        """
        当sam编码完成后调用
        注意ISAT在单独线程对前后图像进行预编码，通过self.mainwindow.seganythread.results_dict中获取编码结果

        Arguments:
            index (int): Index of the image which has been encoded.
        """
        pass

    def on_mouse_move_event(self, scene_pos):
        """
        当鼠标移动时调用

        Arguments:
            scene_pos (QtCore.QPointF): 坐标，限制在图像范围内
        """
        pass

    def on_mouse_press_event(self, scene_pos):
        """
        当鼠标按下时调用

        Arguments:
            scene_pos (QtCore.QPointF): 坐标，限制在图像范围内
        """
        pass

    def on_mouse_release_event(self, scene_pos):
        """
        当鼠标释放时调用

        Arguments:
            scene_pos (QtCore.QPointF): 坐标，限制在图像范围内
        """
        pass

    def on_mouse_pressed_and_mouse_move_event(self, scene_pos):
        """
        当鼠标拖动时调用

        Arguments:
            scene_pos (QtCore.QPointF): 坐标，限制在图像范围内
        """
        pass

    def application_start_event(self):
        """
        软件启动后触发
        """
        pass

    def application_shutdown_event(self):
        """
        软件关闭前触发
        """
        pass
