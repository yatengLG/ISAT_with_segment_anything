# -*- coding: utf-8 -*-
# @Author  : LG

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QThread, pyqtSignal
from ISAT.ui.model_manager_dialog import Ui_Dialog
from ISAT.configs import CHECKPOINT_PATH
from ISAT.segment_any.model_zoo import model_dict
from urllib import request
from functools import partial
import shutil
import time
import os


class DownloadThread(QThread):
    """
    Thread for downloading checkpoint.

    Attributes:
        urls (list): Url list, which include: 'huggingface' and 'modelscope'.
        name (str): Checkpoint name.
        pause (bool): Pause download.
    """
    tag = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super(DownloadThread, self).__init__(parent)
        self.urls = None
        self.name = None
        self.pause = False

        self.block_size = 4096

    def setNameAndUrl(self, name, urls):
        self.urls = urls
        self.name = name
        self.pause = False

    def run(self):
        if self.name is not None and self.urls is not None:
            tmp_root = os.path.join(CHECKPOINT_PATH, 'tmp')
            if not os.path.exists(tmp_root):
                os.mkdir(tmp_root)

            # 寻找最佳下载链接
            best_time = 1e8
            best_url = self.urls[0]
            for url in self.urls:
                try:
                    start_time = time.time()
                    req = request.Request(url, headers={"Range": "bytes=0-10"})
                    request.urlopen(req, timeout=5)
                    cost_time = time.time() - start_time
                except:
                    cost_time = 1e8
                if cost_time < best_time:
                    best_time = cost_time
                    best_url = url

            print('Download {} from {}'.format(self.name,  best_url))
            # 检查缓存
            downloaded_size = 0
            download_tmp = os.path.join(tmp_root, self.name)
            if os.path.exists(download_tmp):
                with open(download_tmp, 'rb') as f:
                    downloaded_size = len(f.read())

            req = request.Request(best_url, headers={"Range": "bytes=0-"})
            try:
                response = request.urlopen(req, timeout=10)
                total_size = int(response.headers['Content-Length'])
            except Exception as e:
                print('When download {} from {}, {}'.format(self.name, best_url, e))
                return
            # 存在缓存
            if downloaded_size != 0:
                # 判断缓存文件是否下载完
                if downloaded_size > total_size:
                    self.move_from_tmp(download_tmp, os.path.join(CHECKPOINT_PATH, self.name))
                    return
                # 断点续传
                content_range = response.headers.get('Content-Range', None)
                if content_range is not None:
                    req = request.Request(best_url, headers={"Range": "bytes={}-".format(downloaded_size)})
                    response = request.urlopen(req)
                    content_range = response.headers.get('Content-Range', None)
                    if content_range is not None:
                        print('Resume downloading: ', content_range)
                else:
                    print('Not supprot resume download.')
                    downloaded_size = 0

            open_mode = 'wb' if downloaded_size == 0 else 'ab'
            with open(download_tmp, open_mode) as f:
                while True:
                    if self.pause:
                        self.tag.emit(-1, -1)
                        return
                    buffer = response.read(self.block_size)
                    if not buffer:
                        break
                    f.write(buffer)
                    downloaded_size += len(buffer)
                    self.tag.emit(downloaded_size, total_size)
            # 下载完成
            self.move_from_tmp(download_tmp, os.path.join(CHECKPOINT_PATH, self.name))

    def move_from_tmp(self, from_path, to_path):
        try:
            shutil.move(from_path, to_path)
            self.tag.emit(-2, -2)

        except Exception as e:
            print('Error when move {} to {}, {}'.format(from_path, to_path, e))


class ModelManagerDialog(QtWidgets.QDialog, Ui_Dialog):
    """Model manager interface."""
    def __init__(self, parent, mainwindow):
        super(ModelManagerDialog, self).__init__(parent)
        self.mainwindow = mainwindow
        self.setupUi(self)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.download_thread_dict = {}
        self.button_group = QtWidgets.QButtonGroup()
        self.button_group.addButton(self.radioButton_ft_model)

        self.init_ui()
        self.pushButton_clear_tmp.clicked.connect(self.clear_tmp)
        self.pushButton_close.clicked.connect(self.close)
        self.pushButton_open_ft_model.clicked.connect(self.load_fine_tuned_model)
        self.radioButton_ft_model.clicked.connect(self.use_fine_tune_model)

    def init_ui(self):
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)

        for i in range(self.gridLayout.count()):
            self.gridLayout.itemAt(i).widget().deleteLater()

        for index, (model_name, info_dict) in enumerate(model_dict.items()):
            url = info_dict.get('url', '')
            memory = info_dict.get('memory', '')
            bf16_memory = info_dict.get('bf16_memory', '')
            params = info_dict.get('params', '')
            image_segment = info_dict.get('image_segment', False)
            video_segment = info_dict.get('video_segment', False)
            # radio
            radio = QtWidgets.QRadioButton()
            radio.setFixedWidth(30)
            radio.toggled.connect(partial(self.mainwindow.init_segment_anything, os.path.join(CHECKPOINT_PATH, model_name)))
            radio.setEnabled(False)
            self.button_group.addButton(radio)
            # image seg
            image_segment_label = QtWidgets.QLabel()
            pixmap = QtGui.QPixmap(":/icon/icons/校验-小_check-small.svg") if image_segment else QtGui.QPixmap(":/icon/icons/关闭-小_close-small.svg")
            image_segment_label.setPixmap(pixmap)
            image_segment_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            image_segment_label.setFixedWidth(80)
            image_segment_label.setStyleSheet('background-color: rgb(240, 240, 240);' if index % 2 else 'background-color: rgb(255, 255, 255);')
            # video seg
            video_segment_label = QtWidgets.QLabel()
            pixmap = QtGui.QPixmap(":/icon/icons/校验-小_check-small.svg") if video_segment else QtGui.QPixmap(":/icon/icons/关闭-小_close-small.svg")
            video_segment_label.setPixmap(pixmap)
            video_segment_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            video_segment_label.setFixedWidth(80)
            video_segment_label.setStyleSheet('background-color: rgb(240, 240, 240);' if index % 2 else 'background-color: rgb(255, 255, 255);')
            # model name
            name_label = QtWidgets.QLabel()
            name_label.setFont(font)
            name_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
            name_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)
            name_label.setText(model_name)
            name_label.setStyleSheet('background-color: rgb(240, 240, 240);' if index % 2 else 'background-color: rgb(255, 255, 255);')
            # 显存占用
            memory_label = QtWidgets.QLabel()
            memory_label.setFont(font)
            memory_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            memory_label.setText(bf16_memory)
            memory_label.setFixedWidth(100)
            memory_label.setStyleSheet('background-color: rgb(240, 240, 240);' if index % 2 else 'background-color: rgb(255, 255, 255);')
            # 权重大小
            params_label = QtWidgets.QLabel()
            params_label.setFont(font)
            params_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            params_label.setText(params)
            params_label.setFixedWidth(100)
            params_label.setStyleSheet('background-color: rgb(240, 240, 240);' if index % 2 else 'background-color: rgb(255, 255, 255);')
            # 下载/删除按钮
            ops_button = QtWidgets.QPushButton()
            ops_button.setFixedWidth(300)
            ops_button.setFixedHeight(30)
            ops_button.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
            ops_button.setFont(font)

            if os.path.exists(os.path.join(CHECKPOINT_PATH, model_name)):
                # ops_button.setStyleSheet('QWidget {background-color: %s}' % 'red')
                ops_button.setStyleSheet('QWidget {color: %s}' % 'red')
                ops_button.setText('delete')
                ops_button.clicked.connect(partial(self.delete, model_name))
                radio.setEnabled(True)
            else:
                # ops_button.setStyleSheet('QWidget {background-color: %s}' % 'green')
                ops_button.setStyleSheet('QWidget {color: %s}' % 'green')
                ops_button.setText('download')
                ops_button.clicked.connect(partial(self.download, model_name))

            self.gridLayout.addWidget(radio, index, 0, 1, 1)
            self.gridLayout.addWidget(image_segment_label, index, 1, 1, 1)
            self.gridLayout.addWidget(video_segment_label, index, 2, 1, 1)
            self.gridLayout.addWidget(name_label, index, 3, 1, 1)
            self.gridLayout.addWidget(memory_label, index, 4, 1, 1)
            self.gridLayout.addWidget(params_label, index, 5, 1, 1)
            self.gridLayout.addWidget(ops_button, index, 6, 1, 1)

    def update_ui(self):
        for index, (model_name, info_dict) in enumerate(model_dict.items()):
            url = info_dict.get('url', '')
            memory = info_dict.get('memory', '')
            bf16_memory = info_dict.get('bf16_memory', '')
            params = info_dict.get('params', '')
            image_segment = info_dict.get('image_segment', False)
            video_segment = info_dict.get('video_segment', False)

            radio = self.gridLayout.itemAtPosition(index, 0).widget()
            image_segment_label = self.gridLayout.itemAtPosition(index, 1).widget()
            video_segment_label = self.gridLayout.itemAtPosition(index, 2).widget()
            name_label = self.gridLayout.itemAtPosition(index, 3).widget()
            memory_label = self.gridLayout.itemAtPosition(index, 4).widget()
            params_label = self.gridLayout.itemAtPosition(index, 5).widget()
            ops_button = self.gridLayout.itemAtPosition(index, 6).widget()

            radio.blockSignals(True)
            if self.mainwindow.use_segment_anything:
                current_model_name = os.path.split(self.mainwindow.segany.checkpoint)[-1]
                radio.setChecked(current_model_name == model_name)
            else:
                radio.setChecked(False)
            radio.blockSignals(False)

            memory_label.setText(bf16_memory if self.mainwindow.cfg['software']['use_bfloat16'] else memory)

            if os.path.exists(os.path.join(CHECKPOINT_PATH, model_name)):
                # ops_button.setStyleSheet('QWidget {background-color: %s}' % 'red')
                ops_button.setStyleSheet('QWidget {color: %s}' % 'red')
                ops_button.setText('delete')
                ops_button.clicked.connect(partial(self.delete, model_name))
                radio.setEnabled(True)
            else:
                # ops_button.setStyleSheet('QWidget {background-color: %s}' % 'green')
                ops_button.setStyleSheet('QWidget {color: %s}' % 'green')
                ops_button.setText('download')
                ops_button.clicked.connect(partial(self.download, model_name))
                radio.setEnabled(False)

    def download(self, model_name):
        button = self.sender()
        button.setText('downloading')
        info_dict = model_dict.get(model_name, {})
        urls = info_dict.get('urls', None)

        if model_name in self.download_thread_dict:
            download_thread = self.download_thread_dict[model_name]
        else:
            download_thread = DownloadThread(self)
            self.download_thread_dict[model_name] = download_thread

        download_thread.setNameAndUrl(model_name, urls)
        download_thread.tag.connect(partial(self.download_process, button, model_name))
        download_thread.start()

        button.setEnabled(False)
        button.clicked.connect(partial(self.pause, model_name))
        button.setEnabled(True)

    def download_process(self, button:QtWidgets.QPushButton, model_name, downloaded_size, total_size):
        if downloaded_size == -1 and total_size == -1:
            button.setText('continue')
            button.setEnabled(False)
            button.clicked.connect(partial(self.download, model_name))
            button.setEnabled(True)

        elif downloaded_size == -2 and total_size == -2:
            button.setText('delete')
            button.setStyleSheet('QWidget {color: %s}' % 'red')
            button.setEnabled(False)
            button.clicked.connect(partial(self.delete, model_name))
            button.setEnabled(True)
            # self.mainwindow.update_menuSAM()
            self.update_ui()
        else:
            button.setText('{:.2f}% - {}/{}M'.format(downloaded_size/total_size*100, downloaded_size//1000000, total_size//1000000))

    def pause(self, model_name):
        download_thread:DownloadThread = self.download_thread_dict[model_name]
        download_thread.pause = True

    def delete(self, model_name):
        button = self.sender()
        try:
            os.remove(os.path.join(CHECKPOINT_PATH, model_name))
            button.setText('download')
            button.setStyleSheet('QWidget {color: %s}' % 'green')
            button.setEnabled(False)
            button.clicked.connect(partial(self.download, model_name))
            button.setEnabled(True)
        except Exception as e:
            print('Error when remove {}, {}'.format(
                os.path.join(CHECKPOINT_PATH, model_name), e))
        self.update_ui()

    def clear_tmp(self):
        remove_list = []
        for name in os.listdir(os.path.join(CHECKPOINT_PATH, 'tmp')):
            tmp_file = os.path.join(CHECKPOINT_PATH, 'tmp', name)
            if name in self.download_thread_dict:
                thread = self.download_thread_dict.get(name, None)
                if thread is not None:
                    if thread.isRunning():
                        continue
            try:
                os.remove(tmp_file)
                remove_list.append(tmp_file)

            except Exception as e:
                print('Error when remove {}, {}'.format(
                    os.path.join(CHECKPOINT_PATH, name), e))
        QtWidgets.QMessageBox.information(
            self,
            'clear tmp',
            'Remove tmp: [' + '],['.join(remove_list) + ']'
        )

    def load_fine_tuned_model(self):
        filter = "Checkpoint (*.pt *.pth *.pkl);;All files (*)"
        path, suffix = QtWidgets.QFileDialog.getOpenFileName(self, caption='Open file', filter=filter)
        if path:
            self.lineEdit_ft_model.setText(path)
            self.radioButton_ft_model.click()

    def use_fine_tune_model(self):
        model_path = self.lineEdit_ft_model.text()
        if model_path == '':
            QtWidgets.QMessageBox.warning(self, 'Warning', 'No checkpoint selected.')
            return

        supported_model_prefix = (
            'mobile_sam',
            'sam_hq_vit_h',
            'sam_hq_vit_l',
            'sam_hq_vit_b',
            'sam_hq_vit_tiny',
            'sam_vit_h',
            'sam_vit_l',
            'sam_vit_b',
            'edge_sam',
            'sam-med2d_b',
            'sam2_hiera_large',
            'sam2_hiera_base_plus',
            'sam2_hiera_small',
            'sam2_hiera_tiny',
            'sam2.1_hiera_large',
            'sam2.1_hiera_base_plus',
            'sam2.1_hiera_small',
            'sam2.1_hiera_tiny',
        )

        if model_path and os.path.basename(model_path).startswith(supported_model_prefix):
            self.mainwindow.init_segment_anything(model_path)
            return True
        else:
            QtWidgets.QMessageBox.warning(self, 'warning',
                                          "Checkpoint's name must start with: \n    {}".format('\n    '.join(supported_model_prefix)))
        return False
