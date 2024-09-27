# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtWidgets, QtCore, QtGui
from ISAT.ui.video_to_frames import Ui_Dialog
import cv2
import os


class Video2FramesDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent, mainwindow):
        super().__init__(parent=parent)
        self.setupUi(self)
        self.mainwindow = mainwindow
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        self.video_path = None
        self.save_root = None
        self.pushButton_video_path.clicked.connect(self.open_file)
        self.pushButton_frames_root.clicked.connect(self.open_dir)
        self.pushButton_start.clicked.connect(self.start)

    def open_file(self):
        file, _ = QtWidgets.QFileDialog.getOpenFileName(self, caption='Open video', filter="mp4 (*.mp4 *.MP4);;All Files (*)")
        if file:
            self.video_path = file
            self.lineEdit_video_path.setText(file)
        else:
            self.video_path = None
            self.lineEdit_video_path.clear()

    def open_dir(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(self, caption='Open dir')
        if dir:
            self.save_root = dir
            self.lineEdit_frames_root.setText(dir)
        else:
            self.save_root = None
            self.lineEdit_frames_root.clear()

    def start(self):
        if self.video_path is None:
            self.textBrowser.append('ERROR | Video path is None!')
            return
        if self.save_root is None:
            self.textBrowser.append('ERROR | Save root is None!')
            return

        video_name: str = os.path.split(self.video_path)[-1]
        video_name_without_suffix = video_name.split('.')[0]

        camera = cv2.VideoCapture(self.video_path)
        frame_count = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))

        self.textBrowser.append('Video path: {}'.format(self.video_path))
        self.textBrowser.append('Save  root: {}'.format(self.save_root))
        self.textBrowser.append('Frames num: {}'.format(frame_count))

        self.progressBar.setMaximum(frame_count)

        for index in range(frame_count):
            try:
                res, image = camera.read()
                cv2.imwrite(os.path.join(self.save_root, video_name_without_suffix + '_{:0>12}.jpg'.format(index)), image)
                self.progressBar.setValue(index+1)
            except Exception as e:
                self.textBrowser.append('ERROR | frame: {} - {}'.format(index+1, e))

        camera.release()

