# -*- coding: utf-8 -*-
# @Author  : LG

import os
from pathlib import Path

import cv2
from PyQt5 import QtCore, QtWidgets

from ISAT.ui.video_to_frames import Ui_Dialog
from ISAT.utils.check import has_chinese


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
        self.pushButton_close.clicked.connect(self.close)

    def open_file(self):
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, caption="Open video", filter="mp4 (*.mp4 *.MP4);;All Files (*)"
        )
        if file:
            self.video_path = file
            self.lineEdit_video_path.setText(file)

            try:
                camera = cv2.VideoCapture(self.video_path)
                fps = int(camera.get(cv2.CAP_PROP_FPS))
                width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_count = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))

                self.spinBox_step.setValue(fps)

                self.textBrowser.append("Frames num   : {}".format(frame_count))
                self.textBrowser.append("Frames width : {}".format(width))
                self.textBrowser.append("Frames height: {}".format(height))
                self.textBrowser.append("Frames fps   : {}".format(fps))
            except Exception as e:
                self.textBrowser.append("ERROR | {}".format(e))

        else:
            self.video_path = None
            self.lineEdit_video_path.clear()

    def open_dir(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(self, caption="Open dir")
        if dir:
            if has_chinese(dir):
                self.textBrowser.append(f"ERROR | Not supported Chinese path. But get {dir}")
                return

            self.save_root = dir
            self.lineEdit_frames_root.setText(dir)
        else:
            self.save_root = None
            self.lineEdit_frames_root.clear()

    def start(self):
        if self.video_path is None:
            self.textBrowser.append("ERROR | Video path is None!")
            return
        if self.save_root is None:
            self.textBrowser.append("ERROR | Save root is None!")
            return

        video_name: str = os.path.split(self.video_path)[-1]
        video_name_without_suffix = video_name.split(".")[0]

        camera = cv2.VideoCapture(self.video_path)
        frame_count = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
        step = self.spinBox_step.value()

        save_indices = list(range(0, frame_count, step))
        self.progressBar.setMaximum(len(save_indices))

        for index, frame_idx in enumerate(save_indices):
            try:
                camera.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # 直接跳转
                res, image = camera.read()
                frame_path = Path(self.save_root) / f"{video_name_without_suffix}_{frame_idx:0>12}.jpg"
                cv2.imwrite(
                    str(frame_path),
                    image,
                )
                self.textBrowser.append(f"Frame {frame_idx:0>12} - {frame_path}")
            except Exception as e:
                self.textBrowser.append(f"ERROR | frame: {frame_idx:0>12} - {e}")
            self.progressBar.setValue(index + 1)
        camera.release()
