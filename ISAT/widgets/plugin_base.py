# -*- coding: utf-8 -*-
# @Author  : LG

from abc import ABC, abstractmethod
import numpy as np


class PluginBase(ABC):
    def __init__(self):
        self.enabled = False

    @abstractmethod
    def init_plugin(self, mainwindow):
        """
        接受主程序调用，进行初始化
        :param mainwindow: 传入mainwindow，便于访问ISAT主程序的各种方法与属性
        """
        pass

    @abstractmethod
    def enable_plugin(self):
        """
        启动插件
        :param mainwindow:
        :return:
        """
        pass

    @abstractmethod
    def disable_plugin(self):
        """
        禁用插件
        :param mainwindow:
        :return:
        """
        pass

    @abstractmethod
    def get_plugin_author(self) -> str:
        """
        获取插件作者
        :return:
        """
        pass

    @abstractmethod
    def get_plugin_version(self) -> str:
        """
        获取插件版本
        :return:
        """
        pass

    @abstractmethod
    def get_plugin_description(self) -> str:
        """
        获取插件描述
        :return:
        """
        pass

    def get_plugin_name(self) -> str:
        """
        获取插件名
        :return:
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
        :param image_path: 图片路径
        :return:
        """
        pass

    def after_image_open_event(self):
        """
        当图片打开后调用
        :param:
        :return:
        """
        pass

    def before_annotation_start_event(self):
        """
        当开始标注前调用
        :return:
        """
        pass

    def after_annotation_created_event(self):
        """
        当标注创建完成后调用
        :return:
        """
        pass

    def after_annotation_changed_event(self):
        """
        当标注发生变化后调用，包括创建新标注，拖动顶点，移动多边形，改变多边形属性等
        :return:
        """
        pass

    def before_annotations_save_event(self):
        """
        当保存标注文件前调用
        :return:
        """
        pass

    def after_annotations_saved_event(self):
        """
        当保存标注文件后调用
        :return:
        """
        pass

    def after_sam_encode_finished_event(self, index):
        """
        当sam编码完成后调用
        注意ISAT在单独线程对前后图像进行预编码，通过self.mainwindow.seganythread.results_dict中获取编码结果
        :param index: 编码完成的图片index
        :return:
        """
        pass


    def on_mouse_move_event(self, scene_pos):
        """
        当鼠标移动时调用
        :param scene_pos: 坐标，限制在图像范围内
        :return:
        """
        pass

    def on_mouse_press_event(self, scene_pos):
        """
        当鼠标按下时调用
        :param scene_pos: 坐标，限制在图像范围内
        :return:
        """
        pass

    def on_mouse_release_event(self, scene_pos):
        """
        当鼠标释放时调用
        :param scene_pos: 坐标，限制在图像范围内
        :return:
        """
        pass

    def on_mouse_pressed_and_mouse_move_event(self, scene_pos):
        """
        当鼠标拖动时调用
        :param scene_pos: 坐标，限制在图像范围内
        :return:
        """
        pass

    def application_start_event(self):
        """
        软件启动后触发
        :return:
        """
        pass

    def application_shutdown_event(self):
        """
        软件关闭前触发
        :return:
        """
        pass
