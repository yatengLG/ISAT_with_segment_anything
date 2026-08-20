# -*- coding: utf-8 -*-
# @Author  : LG

import torch

from PyQt5 import QtCore, QtGui, QtWidgets

from ISAT.ui.setting_dialog import Ui_Dialog


class SettingDialog(QtWidgets.QDialog, Ui_Dialog):
    """Software settings dialog (self-contained).

    Loads its widget states from ``mainwindow.cfg`` via :meth:`load_from_cfg`
    and writes every change straight back to the config — the main window
    never pokes the widgets of this dialog directly.
    """

    def __init__(self, parent, mainwindow):
        QtWidgets.QDialog.__init__(self, parent)
        self.mainwindow = mainwindow
        self.setupUi(self)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        self.checkBox_auto_save.stateChanged.connect(
            self.mainwindow.change_auto_save_state
        )
        self.checkBox_real_time_area.stateChanged.connect(
            self.mainwindow.change_real_time_area_state
        )
        self.checkBox_approx_polygon.stateChanged.connect(
            self.mainwindow.change_approx_polygon_state
        )
        self.checkBox_polygon_invisible.stateChanged.connect(
            self.mainwindow.change_create_mode_invisible_polygon_state
        )
        self.checkBox_show_edge.stateChanged.connect(self.mainwindow.change_edge_state)
        self.checkBox_show_prompt.stateChanged.connect(
            self.mainwindow.change_prompt_visiable
        )
        self.checkBox_use_bfloat16.stateChanged.connect(
            self.mainwindow.change_bfloat16_state
        )
        self.checkBox_use_video_segmentation.stateChanged.connect(
            self.mainwindow.change_use_video_segmentation_state
        )
        # 滑块：先更新自身标签，再通知 mainwindow 应用（mainwindow 不再回写本对话框）
        self.horizontalSlider_mask_alpha.valueChanged.connect(
            self._mask_alpha_changed
        )
        self.horizontalSlider_polygon_alpha_hover.valueChanged.connect(
            self._polygon_alpha_hover_changed
        )
        self.horizontalSlider_polygon_alpha_no_hover.valueChanged.connect(
            self._polygon_alpha_no_hover_changed
        )
        self.horizontalSlider_vertex_size.valueChanged.connect(
            self._vertex_size_changed
        )
        self.comboBox_contour_mode.currentIndexChanged.connect(
            self.contour_mode_index_changed
        )
        self.comboBox_contour_method.currentIndexChanged.connect(
            self.contour_method_index_changed
        )
        self.pushButton_close.clicked.connect(self.close)

    # ------------------------------------------------------------------
    #  Slider handlers — update the dialog's own labels, then apply.
    # ------------------------------------------------------------------

    def _mask_alpha_changed(self, value: int):
        self.label_mask_alpha.setText("{}".format(value / 10))
        self.mainwindow.change_mask_alpha(value)

    def _polygon_alpha_hover_changed(self, value: int):
        self.label_polygon_alpha_hover.setText("{}".format(value / 10))
        self.mainwindow.change_polygon_alpha_hover(value)

    def _polygon_alpha_no_hover_changed(self, value: int):
        self.label_polygon_alpha_no_hover.setText("{}".format(value / 10))
        self.mainwindow.change_polygon_alpha_no_hover(value)

    def _vertex_size_changed(self, value: int):
        self.label_vertex_size.setText("{}".format(value))
        self.mainwindow.change_vertex_size(value)

    def contour_mode_index_changed(self, index):
        if index == 0:
            contour_mode = "external"
        elif index == 1:
            contour_mode = "max_only"
        else:
            contour_mode = "all"
        self.mainwindow.change_contour_mode(contour_mode)

    def contour_method_index_changed(self, index):
        if index == 0:
            contour_method = "SIMPLE"
        elif index == 1:
            contour_method = "TC89_KCOS"
        else:
            contour_method = "NONE"
        self.mainwindow.change_contour_method(contour_method)

    # ------------------------------------------------------------------
    #  Load current values from cfg.
    # ------------------------------------------------------------------

    def load_from_cfg(self):
        """Refresh all widget states from ``mainwindow.cfg["software"]``.

        Signals are blocked while the widgets are being set, otherwise every
        ``setChecked`` / ``setValue`` / ``setCurrentIndex`` would fire the
        change handlers wired in ``__init__`` (model re-init, image reload,
        config save) — a full side-effect sweep on every dialog open.
        """
        cfg = self.mainwindow.cfg["software"]

        # 加载期间阻断信号，避免 set* 触发 change_* 副作用
        widgets = (
            self.checkBox_auto_save,
            self.checkBox_real_time_area,
            self.checkBox_approx_polygon,
            self.checkBox_polygon_invisible,
            self.checkBox_show_edge,
            self.checkBox_show_prompt,
            self.checkBox_use_bfloat16,
            self.checkBox_use_video_segmentation,
            self.horizontalSlider_mask_alpha,
            self.horizontalSlider_polygon_alpha_hover,
            self.horizontalSlider_polygon_alpha_no_hover,
            self.horizontalSlider_vertex_size,
            self.comboBox_contour_mode,
            self.comboBox_contour_method,
        )
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self.checkBox_auto_save.setChecked(cfg.get("auto_save", False))
            self.checkBox_real_time_area.setChecked(cfg.get("real_time_area", False))
            self.checkBox_approx_polygon.setChecked(cfg.get("use_polydp", True))
            self.checkBox_polygon_invisible.setChecked(
                cfg.get("create_mode_invisible_polygon", True)
            )
            self.checkBox_show_edge.setChecked(cfg.get("show_edge", True))
            self.checkBox_show_prompt.setChecked(cfg.get("show_prompt", False))

            self.checkBox_use_bfloat16.setChecked(cfg.get("use_bfloat16", False))
            self.checkBox_use_bfloat16.setEnabled(torch.cuda.is_available())
            self.checkBox_use_video_segmentation.setChecked(
                cfg.get("use_video_segmentation", True)
            )

            mask_alpha = cfg.get("mask_alpha", 0.5)
            self.horizontalSlider_mask_alpha.setValue(int(mask_alpha * 10))
            self.label_mask_alpha.setText("{}".format(mask_alpha))

            polygon_alpha_hover = cfg.get("polygon_alpha_hover", 0.6)
            self.horizontalSlider_polygon_alpha_hover.setValue(
                int(polygon_alpha_hover * 10)
            )
            self.label_polygon_alpha_hover.setText("{}".format(polygon_alpha_hover))

            polygon_alpha_no_hover = cfg.get("polygon_alpha_no_hover", 0.3)
            self.horizontalSlider_polygon_alpha_no_hover.setValue(
                int(polygon_alpha_no_hover * 10)
            )
            self.label_polygon_alpha_no_hover.setText("{}".format(polygon_alpha_no_hover))

            vertex_size = cfg.get("vertex_size", 1)
            self.horizontalSlider_vertex_size.setValue(int(vertex_size))
            self.label_vertex_size.setText("{}".format(int(vertex_size)))

            contour_mode = cfg.get("contour_mode", "max_only")
            self.comboBox_contour_mode.setCurrentIndex(
                {"external": 0, "max_only": 1, "all": 2}.get(contour_mode, 0)
            )

            contour_method = cfg.get("contour_method", "SIMPLE")
            self.comboBox_contour_method.setCurrentIndex(
                {"SIMPLE": 0, "TC89_KCOS": 1, "NONE": 2}.get(contour_method, 0)
            )
        finally:
            for widget in widgets:
                widget.blockSignals(False)
