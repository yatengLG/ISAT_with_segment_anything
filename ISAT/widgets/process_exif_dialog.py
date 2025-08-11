# -*- coding: utf-8 -*-
# @Author  : LG

import os
from ISAT.ui.process_exif_dialog import Ui_Dialog
from PyQt5 import QtWidgets, QtCore
from PIL import Image, ImageOps


class ProcessExifDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent, mainwindow):
        super().__init__(parent=parent)
        self.setupUi(self)
        self.mainwindow = mainwindow

        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        self.radioButton_apply_exif.setChecked(True)
        self.pushButton_image_root.clicked.connect(self.open_dir)
        self.pushButton_save_root.clicked.connect(self.open_dir)
        self.pushButton_start.clicked.connect(self.start)
        self.pushButton_close.clicked.connect(self.close)

    def open_dir(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(self, caption='Open dir')
        if self.sender() == self.pushButton_image_root:
            lineEdit = self.lineEdit_image_root
        elif self.sender() == self.pushButton_save_root:
            lineEdit = self.lineEdit_save_root
        else:
            return
        if dir:
            lineEdit.setText(dir)
        else:
            lineEdit.clear()

    def start(self):
        self.pushButton_start.setEnabled(False)
        self.textBrowser.clear()
        self.progressBar.reset()

        image_root = self.lineEdit_image_root.text()
        save_root = self.lineEdit_save_root.text()
        apply_exif = self.radioButton_apply_exif.isChecked()

        if not (image_root or save_root):
            self.pushButton_start.setEnabled(True)
            return

        images = os.listdir(image_root)
        self.progressBar.setMaximum(len(images))

        for index, image_name in enumerate(images):
            self.textBrowser.append('{:>8d} - Processing - {}'.format(index, image_name))
            try:
                image = Image.open(os.path.join(image_root, image_name))
            except:
                self.textBrowser.append('{} Invalid image.'.format(' '*10))
                image = None

            if image is not None:
                exif_info = image.getexif()
                if exif_info and exif_info.get(274, 1) != 1:
                    self.textBrowser.append('{} Has rotation tag.'.format(' ' * 10, ))
                    if apply_exif:
                        self.textBrowser.append('{} Ori size    : w-{} h-{}.'.format(' ' * 10, image.width, image.height))
                        image = ImageOps.exif_transpose(image)
                        self.textBrowser.append('{} Rotated size: w-{} h-{}.'.format(' ' * 10, image.width, image.height))
                    image.save(os.path.join(save_root, image_name))
                    self.textBrowser.append('{} Resave finished'.format(' '* 10))
                else:
                    self.textBrowser.append('{} No rotation tag.'.format(' '*10))

            self.progressBar.setValue(index + 1)

        self.pushButton_start.setEnabled(True)

