# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtCore, QtGui, QtWidgets
from ISAT.ui.setting_dialog import Ui_Dialog

class SettingDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent, mainwindow):
        QtWidgets.QDialog.__init__(self, parent)
        self.mainwindow = mainwindow
        self.setupUi(self)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        self.checkBox_auto_save.stateChanged.connect(self.mainwindow.change_auto_save_state)
        self.checkBox_real_time_area.stateChanged.connect(self.mainwindow.change_real_time_area_state)
        self.checkBox_approx_polygon.stateChanged.connect(self.mainwindow.change_approx_polygon_state)
        self.checkBox_polygon_invisible.stateChanged.connect(self.mainwindow.change_create_mode_invisible_polygon_state)
        self.checkBox_show_edge.stateChanged.connect(self.mainwindow.change_edge_state)
        self.checkBox_show_prompt.stateChanged.connect(self.mainwindow.change_prompt_visiable)
        self.checkBox_use_bfloat16.stateChanged.connect(self.mainwindow.change_bfloat16_state)
        self.horizontalSlider_vertex_size.valueChanged.connect(self.mainwindow.change_vertex_size)
        self.horizontalSlider_mask_alpha.valueChanged.connect(self.mainwindow.change_mask_alpha)
        self.comboBox_contour_mode.currentIndexChanged.connect(self.contour_mode_index_changed)
        self.pushButton_close.clicked.connect(self.close)

    def contour_mode_index_changed(self, index):
        if index == 0:
            contour_mode = 'external'
        elif index == 1:
            contour_mode = 'max_only'
        else:
            contour_mode = 'all'
        self.mainwindow.change_contour_mode(contour_mode)