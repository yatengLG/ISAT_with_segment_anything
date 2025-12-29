# -*- coding: utf-8 -*-
# @Author  : LG

from functools import partial
from PyQt5 import QtWidgets

from ISAT.ui.visual_prompt_dock import Ui_Form


class VisualPromptDockWidget(QtWidgets.QWidget, Ui_Form):
    def __init__(self, mainwindow):
        super(VisualPromptDockWidget, self).__init__()
        self.setupUi(self)
        self.mainwindow = mainwindow

        self.pushButton_positive_box.clicked.connect(partial(self.mainwindow.scene.start_segment_anything_visual, True))
        self.pushButton_negative_box.clicked.connect(partial(self.mainwindow.scene.start_segment_anything_visual, False))
        self.pushButton_predict.clicked.connect(self._predict)

    def _predict(self):
        if (not self.mainwindow.use_segment_anything) or self.mainwindow.segany.model_source != "sam3":
            QtWidgets.QMessageBox.warning(self, "warning", "Only SAM3 is supported!")
            self.mainwindow.scene.cancel_draw()
            return

        if len(self.mainwindow.scene.prompt_visual_items) < 1:
            self.mainwindow.scene.cancel_draw()
            return

        self.mainwindow.setEnabled(False)
        self.label_num_objects.setText("......")
        self.label_num_objects.repaint()

        prompt_visual_positions = [[visual_item.points[0].x(),
                                    visual_item.points[0].y(),
                                    visual_item.points[1].x(),
                                    visual_item.points[1].y()]
                                   for visual_item in self.mainwindow.scene.prompt_visual_items]

        num_masks = self.mainwindow.predict_current_image_with_visual_prompt(
            self.mainwindow.current_category,
            prompt_visual_positions,
            self.mainwindow.scene.prompt_visual_labels
        )

        self.mainwindow.scene.finish_draw()

        self.label_num_objects.setText(f"{num_masks}")
        self.mainwindow.setEnabled(True)
