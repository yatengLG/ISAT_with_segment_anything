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

        self.pushButton_predict.clicked.connect(self.predict)

    def predict(self):
        self.mainwindow.predict_current_image_with_text_prompt(self.lineEdit_prompt_text.text())