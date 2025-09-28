# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtWidgets, QtCore
from ISAT.ui.about_dialog import Ui_Dialog


class AboutDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent):
        super(AboutDialog, self).__init__(parent)
        self.setupUi(self)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        from ISAT import __version__

        version_text = "V{}".format(__version__)
        self.label_version.setText(version_text)
        self.pushButton_close.clicked.connect(self.close)
