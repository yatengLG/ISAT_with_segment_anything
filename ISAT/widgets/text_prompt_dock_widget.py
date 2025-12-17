# -*- coding: utf-8 -*-
# @Author  : LG

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
        categories = [self.lineEdit_prompt_text_for_single_category.text()]
        self._predict(categories)

    def predict_for_multi_categories(self):
        categories = [category for category, _ in self.mainwindow.category_color_dict.items()]
        self._predict(categories)

    def _predict(self, categories: list):
        if (not self.mainwindow.use_segment_anything) or self.mainwindow.segany.model_source != "sam3":
            QtWidgets.QMessageBox.warning(self, "warning", "Only SAM3 is supported!")
            return

        self.mainwindow.setEnabled(False)
        self.label_num_objects.setText("......")
        self.label_num_objects.repaint()
        num_masks_all = 0

        self.progressBar.setMaximum(len(categories))
        self.progressBar.setValue(0)

        for index, category in enumerate(categories):
            if category == "__background__":
                continue
            num_masks = self.mainwindow.predict_current_image_with_text_prompt(category)
            if num_masks is None:
                continue
            num_masks_all += num_masks
            self.progressBar.setValue(index + 1)

        self.label_num_objects.setText(f"{num_masks_all}")

        self.mainwindow.setEnabled(True)