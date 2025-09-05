# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtWidgets, QtCore, QtGui
from ISAT.ui.file_dock import Ui_Form
import os


class FilesDockWidget(QtWidgets.QWidget, Ui_Form):
    def __init__(self, mainwindow):
        super(FilesDockWidget, self).__init__()
        self.setupUi(self)
        self.mainwindow = mainwindow
        self.listWidget.clicked.connect(self.listwidget_doubleclick)
        self.lineEdit_jump.returnPressed.connect(self.mainwindow.jump_to)

        self.setAcceptDrops(True)

    def update_widget(self):
        """Update the file list widget."""
        self.listWidget.clear()
        if self.mainwindow.files_list is None:
            return

        for idx, file_name in enumerate(self.mainwindow.files_list):
            # _, file_name = os.path.split(file_path)
            item = QtWidgets.QListWidgetItem()
            item.setSizeHint(QtCore.QSize(200, 30))
            # item, item_widget = self.generate_item_and_itemwidget(file_name)

            item.setText(f'[{idx + 1}] {file_name}')
            self.listWidget.addItem(item)
            # self.listWidget.setItemWidget(item, item_widget)

        self.label_all.setText('{}'.format(len(self.mainwindow.files_list)))

    def set_select(self, row):
        """Set the current file highlight."""
        self.listWidget.setCurrentRow(row)

    def listwidget_doubleclick(self):
        """Double-click the file list item to open the image."""
        row = self.listWidget.currentRow()
        self.mainwindow.show_image(row)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Support drop event for open a image file or image directory."""
        if len(event.mimeData().urls()) != 1:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Only support one path or dir.')
            return
        # 这里与mainwindow.opend_dir逻辑一致
        path = event.mimeData().urls()[0].toLocalFile()
        if os.path.isdir(path):
            dir = path
            # 等待sam线程退出，并清空特征缓存
            if self.mainwindow.use_segment_anything:
                self.mainwindow.seganythread.wait()
                self.mainwindow.seganythread.results_dict.clear()

            self.mainwindow.files_list.clear()
            self.mainwindow.files_dock_widget.listWidget.clear()

            files = []
            suffixs = tuple(
                ['{}'.format(fmt.data().decode('ascii').lower()) for fmt in QtGui.QImageReader.supportedImageFormats()])
            for f in os.listdir(dir):
                if f.lower().endswith(suffixs) or f.lower().endswith('.dcm'):
                    # f = os.path.join(dir, f)
                    files.append(f)
            files = sorted(files)
            self.mainwindow.files_list = files

            self.mainwindow.files_dock_widget.update_widget()

            self.mainwindow.current_index = 0

            self.mainwindow.image_root = dir
            self.mainwindow.actionImages_dir.setStatusTip("Image root: {}".format(self.mainwindow.image_root))

            self.mainwindow.label_root = dir
            self.mainwindow.actionLabel_dir.setStatusTip("Label root: {}".format(self.mainwindow.label_root))

            if os.path.exists(os.path.join(dir, 'isat.yaml')):
                # load setting yaml
                self.mainwindow.config_file = os.path.join(dir, 'isat.yaml')
                self.mainwindow.reload_cfg()

            self.mainwindow.show_image(self.mainwindow.current_index)

        if os.path.isfile(path):
            # 等待sam线程退出，并清空特征缓存
            if self.mainwindow.use_segment_anything:
                self.mainwindow.seganythread.wait()
                self.mainwindow.seganythread.results_dict.clear()

            self.mainwindow.files_list.clear()
            self.mainwindow.files_dock_widget.listWidget.clear()

            suffixs = tuple(
                ['{}'.format(fmt.data().decode('ascii').lower()) for fmt in QtGui.QImageReader.supportedImageFormats()])

            dir, file = os.path.split(path)
            files = []
            if path.lower().endswith(suffixs) or path.lower().endswith('.dcm'):
                files = [file]

            self.mainwindow.files_list = files

            self.mainwindow.files_dock_widget.update_widget()

            self.mainwindow.current_index = 0

            self.mainwindow.image_root = dir
            self.mainwindow.actionImages_dir.setStatusTip("Image root: {}".format(self.mainwindow.image_root))

            self.mainwindow.label_root = dir
            self.mainwindow.actionLabel_dir.setStatusTip("Label root: {}".format(self.mainwindow.label_root))

            if os.path.exists(os.path.join(dir, 'isat.yaml')):
                # load setting yaml
                self.mainwindow.config_file = os.path.join(dir, 'isat.yaml')
                self.mainwindow.reload_cfg()

            self.mainwindow.show_image(self.mainwindow.current_index)
