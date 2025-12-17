# -*- coding: utf-8 -*-
# @Author  : LG

import functools

from PyQt5 import QtWidgets

from ISAT.ui.text_prompt_dock import Ui_Form


class TextPromptDockWidget(QtWidgets.QWidget, Ui_Form):
    def __init__(self, mainwindow):
        super(TextPromptDockWidget, self).__init__()
        self.setupUi(self)
        self.mainwindow = mainwindow

        self.pushButton_predict_for_single_category.clicked.connect(self.predict_for_single_category)
        self.pushButton_catgories_setting.clicked.connect(self.mainwindow.category_setting)
        self.pushButton_predict_for_multi_categoriies.clicked.connect(self.predict_for_multi_categories)

    def predict_for_single_category(self):
        self.label_num_objects.setText("......")
        self.label_num_objects.repaint()
        num_masks = self.mainwindow.predict_current_image_with_text_prompt(
            self.lineEdit_prompt_text_for_single_category.text()
        )
        self.label_num_objects.setText(f"{num_masks}")

    def predict_for_multi_categories(self):
        self.label_num_objects.setText("......")
        self.label_num_objects.repaint()
        num_masks_all = 0
        for category, _ in self.mainwindow.category_color_dict.items():
            if category == "__background__":
                continue
            num_masks = self.mainwindow.predict_current_image_with_text_prompt(category)
            num_masks_all += num_masks
        self.label_num_objects.setText(f"{num_masks_all}")
